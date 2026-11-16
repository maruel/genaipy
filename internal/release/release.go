// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package release installs release archives verified by published SHA-256.
package release

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// Install returns exe inside dir, downloading the archive at u and installing
// its root directory as dir unless a complete install is present.
//
// root is the slash-separated archive directory to install, empty for the
// archive's top level. exe is slash-separated and relative to root. The
// archive must match the SHA-256 published beside it as u+".sha256".
// Extraction stages beside dir and renames the complete tree into place, so an
// interrupted install leaves no directory that looks installed. A concurrent
// installer may publish first; its verified tree is kept.
func Install(ctx context.Context, client *http.Client, u, root, dir, exe string) (string, error) {
	name := path.Base(u)
	exePath := filepath.Join(dir, filepath.FromSlash(exe))
	// The marker holds the verified digest and is written last.
	marker := filepath.Join(dir, ".sha256")
	if installed(marker, exePath) {
		return exePath, nil
	}
	digest, err := publishedDigest(ctx, client, u+".sha256")
	if err != nil {
		return "", err
	}
	parent := filepath.Dir(dir)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return "", err
	}
	archive, err := os.CreateTemp(parent, ".download-*")
	if err != nil {
		return "", err
	}
	defer func() { _ = os.Remove(archive.Name()) }()
	slog.InfoContext(ctx, "release: downloading", "url", u)
	err = download(ctx, client, u, digest, archive)
	if err2 := archive.Close(); err == nil {
		err = err2
	}
	if err != nil {
		return "", err
	}
	staging, err := os.MkdirTemp(parent, ".install-*")
	if err != nil {
		return "", err
	}
	defer func() { _ = os.RemoveAll(staging) }()
	if strings.HasSuffix(name, ".zip") {
		err = extractZip(archive.Name(), staging)
	} else {
		err = extractTarGz(archive.Name(), staging)
	}
	if err != nil {
		return "", fmt.Errorf("extract %s: %w", name, err)
	}
	tree := filepath.Join(staging, filepath.FromSlash(root))
	if fi, err := os.Stat(filepath.Join(tree, filepath.FromSlash(exe))); err != nil || !fi.Mode().IsRegular() {
		return "", fmt.Errorf("%s has no %s", name, path.Join(root, exe))
	}
	if err := os.WriteFile(filepath.Join(tree, ".sha256"), []byte(digest), 0o644); err != nil {
		return "", err
	}
	// Keep a tree a concurrent installer published meanwhile: Windows cannot
	// remove its running executable. Otherwise replace an incomplete tree.
	if installed(marker, exePath) {
		return exePath, nil
	}
	if err := os.RemoveAll(dir); err != nil {
		return "", err
	}
	if err := os.Rename(tree, dir); err != nil && !installed(marker, exePath) {
		return "", err
	}
	return exePath, nil
}

// installed reports whether the tree of marker is complete.
func installed(marker, exe string) bool {
	if _, err := os.Stat(marker); err != nil {
		return false
	}
	fi, err := os.Stat(exe)
	return err == nil && fi.Mode().IsRegular()
}

// publishedDigest reads a "<sha256>  <name>" checksum file. A missing file
// means the release has no such build.
func publishedDigest(ctx context.Context, client *http.Client, u string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, http.NoBody)
	if err != nil {
		return "", err
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("download %s: %w", u, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode == http.StatusNotFound {
		return "", fmt.Errorf("release has no %s", strings.TrimSuffix(path.Base(u), ".sha256"))
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: HTTP %s", u, resp.Status)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, 1024))
	if err != nil {
		return "", fmt.Errorf("download %s: %w", u, err)
	}
	digest, _, _ := strings.Cut(string(b), " ")
	if _, err := hex.DecodeString(digest); err != nil || len(digest) != 2*sha256.Size {
		return "", fmt.Errorf("download %s: malformed checksum", u)
	}
	return digest, nil
}

func download(ctx context.Context, client *http.Client, u, digest string, w io.Writer) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, http.NoBody)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", u, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %s", u, resp.Status)
	}
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(w, h), resp.Body); err != nil {
		return fmt.Errorf("download %s: %w", u, err)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != digest {
		return fmt.Errorf("download %s: SHA-256 %s, want %s", u, got, digest)
	}
	return nil
}

func extractTarGz(name, dst string) error {
	f, err := os.Open(name)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		target, err := entryPath(dst, hdr.Name)
		if err != nil {
			return err
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			err = os.MkdirAll(target, 0o755)
		case tar.TypeReg:
			err = writeFile(target, fs.FileMode(hdr.Mode), tr)
		case tar.TypeSymlink:
			// Shared library aliases point at a sibling file.
			if l := hdr.Linkname; l != filepath.Base(l) || l == "." || l == ".." {
				return fmt.Errorf("symlink %s escapes its directory: %s", hdr.Name, hdr.Linkname)
			}
			if err = os.MkdirAll(filepath.Dir(target), 0o755); err == nil {
				err = os.Symlink(hdr.Linkname, target)
			}
		default:
			return fmt.Errorf("unsupported archive entry %s of type %q", hdr.Name, hdr.Typeflag)
		}
		if err != nil {
			return err
		}
	}
}

func extractZip(name, dst string) error {
	zr, err := zip.OpenReader(name)
	if err != nil {
		return err
	}
	defer func() { _ = zr.Close() }()
	for _, f := range zr.File {
		target, err := entryPath(dst, f.Name)
		if err != nil {
			return err
		}
		mode := f.Mode()
		switch {
		case mode.IsDir():
			err = os.MkdirAll(target, 0o755)
		case mode.IsRegular():
			var r io.ReadCloser
			if r, err = f.Open(); err == nil {
				err = errors.Join(writeFile(target, mode, r), r.Close())
			}
		default:
			return fmt.Errorf("unsupported archive entry %s of mode %s", f.Name, mode)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// entryPath maps an archive entry name to a path inside dst.
func entryPath(dst, name string) (string, error) {
	if !filepath.IsLocal(filepath.FromSlash(name)) {
		return "", fmt.Errorf("archive entry %q escapes the destination", name)
	}
	return filepath.Join(dst, filepath.FromSlash(name)), nil
}

func writeFile(target string, mode fs.FileMode, r io.Reader) error {
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}
	f, err := os.OpenFile(target, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode.Perm()|0o600)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, r)
	return errors.Join(err, f.Close())
}
