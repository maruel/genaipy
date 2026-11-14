// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Pinned NeMo-Speech.cpp release download, verification and extraction.

package nemospeechsrv

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
	"runtime"
	"strings"
)

// Version is the pinned NVIDIA/NeMo-Speech.cpp release.
const Version = "0.3.0"

// Backend selects a NeMo-Speech.cpp release build.
//
// Each platform publishes a subset: Linux x86-64 and Windows x86-64 have CPU,
// CUDA and Vulkan; Linux arm64 has CPU, CUDA12, CUDA13 and Vulkan; macOS arm64
// has CPU and Metal; macOS x86-64 has CPU; Windows arm64 has CUDA. New fails
// for a build the release does not publish.
type Backend string

// Release builds.
const (
	CPU    Backend = "cpu"
	CUDA   Backend = "cuda"
	CUDA12 Backend = "cuda12"
	CUDA13 Backend = "cuda13"
	Metal  Backend = "metal"
	Vulkan Backend = "vulkan"
)

// release returns the archive URL of backend for this platform. An empty
// backend selects Metal on macOS arm64, CUDA on Windows arm64 (its only
// build), and CPU elsewhere.
func release(backend Backend) (string, error) {
	osName := map[string]string{"darwin": "macos", "linux": "linux", "windows": "windows"}[runtime.GOOS]
	arch := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[runtime.GOARCH]
	if osName == "" || arch == "" {
		return "", fmt.Errorf("NeMo-Speech.cpp has no build for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	switch backend {
	case CPU, CUDA, CUDA12, CUDA13, Metal, Vulkan:
	case "":
		switch {
		case runtime.GOOS == "darwin" && runtime.GOARCH == "arm64":
			backend = Metal
		case runtime.GOOS == "windows" && runtime.GOARCH == "arm64":
			backend = CUDA
		default:
			backend = CPU
		}
	default:
		return "", fmt.Errorf("unknown NeMo-Speech.cpp backend %q", backend)
	}
	ext := ".tar.gz"
	if runtime.GOOS == "windows" {
		ext = ".zip"
	}
	return "https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v" + Version + "/nemo-speech-" + Version + "-" + osName + "-" + arch + "-" + string(backend) + ext, nil
}

// install returns the nemo-speech executable of the archive at u, downloading
// and extracting it into cacheDir unless an installed copy is present.
//
// The archive must match the SHA-256 published beside it as u+".sha256" and
// hold one top-level directory named after the archive, with the executable
// in bin. Extraction stages into cacheDir and renames the complete tree into
// place, so an interrupted install leaves no directory that looks installed.
func install(ctx context.Context, client *http.Client, cacheDir, u string) (string, error) {
	name := path.Base(u)
	stem := strings.TrimSuffix(strings.TrimSuffix(name, ".tar.gz"), ".zip")
	dir := filepath.Join(cacheDir, stem)
	exeName := "nemo-speech"
	if runtime.GOOS == "windows" {
		exeName += ".exe"
	}
	exe := filepath.Join(dir, "bin", exeName)
	// The marker holds the verified digest and is written last.
	marker := filepath.Join(dir, ".sha256")
	if installed(marker, exe) {
		return exe, nil
	}
	digest, err := publishedDigest(ctx, client, u+".sha256")
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}
	archive, err := os.CreateTemp(cacheDir, ".download-*")
	if err != nil {
		return "", err
	}
	defer func() { _ = os.Remove(archive.Name()) }()
	slog.InfoContext(ctx, "nemospeechsrv: downloading", "url", u)
	err = download(ctx, client, u, digest, archive)
	if err2 := archive.Close(); err == nil {
		err = err2
	}
	if err != nil {
		return "", err
	}
	staging, err := os.MkdirTemp(cacheDir, ".nemo-speech-*")
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
	tree := filepath.Join(staging, stem)
	if fi, err := os.Stat(filepath.Join(tree, "bin", exeName)); err != nil || !fi.Mode().IsRegular() {
		return "", fmt.Errorf("%s has no bin/%s", name, exeName)
	}
	if err := os.WriteFile(filepath.Join(tree, ".sha256"), []byte(digest), 0o644); err != nil {
		return "", err
	}
	// Replace an incomplete tree. A concurrent installer may win the rename;
	// its verified tree is equally valid.
	if err := os.RemoveAll(dir); err != nil {
		return "", err
	}
	if err := os.Rename(tree, dir); err != nil && !installed(marker, exe) {
		return "", err
	}
	return exe, nil
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
		return "", fmt.Errorf("download NeMo-Speech.cpp checksum: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode == http.StatusNotFound {
		return "", fmt.Errorf("NeMo-Speech.cpp %s has no build %s", Version, strings.TrimSuffix(path.Base(u), ".sha256"))
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
		return fmt.Errorf("download NeMo-Speech.cpp: %w", err)
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
