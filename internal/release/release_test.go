// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for release download verification, archive safety and install reuse.

package release

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
)

type entry struct {
	name, link, body string
}

func tarGz(t *testing.T, entries ...entry) []byte {
	var b bytes.Buffer
	gz := gzip.NewWriter(&b)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		hdr := &tar.Header{Name: e.name, Mode: 0o755, Size: int64(len(e.body)), Typeflag: tar.TypeReg}
		switch {
		case e.link != "":
			hdr.Typeflag, hdr.Linkname, hdr.Size = tar.TypeSymlink, e.link, 0
		case strings.HasSuffix(e.name, "/"):
			hdr.Typeflag = tar.TypeDir
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(e.body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gz.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func zipArchive(t *testing.T, entries ...entry) []byte {
	var b bytes.Buffer
	zw := zip.NewWriter(&b)
	for _, e := range entries {
		w, err := zw.Create(e.name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(e.body)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func digestOf(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

// serve publishes archive at /<name> with its checksum file, and counts
// requests.
func serve(t *testing.T, name string, archive []byte, digest string) (string, *atomic.Int32) {
	var n atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		switch r.URL.Path {
		case "/" + name:
			_, _ = w.Write(archive)
		case "/" + name + ".sha256":
			_, _ = w.Write([]byte(digest + "  " + name + "\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv.URL + "/" + name, &n
}

func TestInstall(t *testing.T) {
	t.Parallel()
	t.Run("valid", func(t *testing.T) {
		t.Parallel()
		t.Run("tar.gz reuse", func(t *testing.T) {
			t.Parallel()
			archive := tarGz(t,
				entry{name: "ns-1/"},
				entry{name: "ns-1/bin/tool", body: "exe"},
				entry{name: "ns-1/lib/libx.so.1.2", body: "lib"},
				entry{name: "ns-1/lib/libx.so.1", link: "libx.so.1.2"})
			u, downloads := serve(t, "ns-1.tar.gz", archive, digestOf(archive))
			cache := t.TempDir()
			for range 2 {
				exe, err := Install(t.Context(), http.DefaultClient, u, "ns-1", filepath.Join(cache, "ns-1"), "bin/tool")
				if err != nil {
					t.Fatal(err)
				}
				if want := filepath.Join(cache, "ns-1", "bin", "tool"); exe != want {
					t.Fatalf("exe = %q, want %q", exe, want)
				}
			}
			// The checksum and the archive, once.
			if n := downloads.Load(); n != 2 {
				t.Errorf("requests = %d, want 2", n)
			}
			if b, err := os.ReadFile(filepath.Join(cache, "ns-1", "lib", "libx.so.1")); err != nil || string(b) != "lib" {
				t.Errorf("library alias = %q, %v", b, err)
			}
			if fi, err := os.Stat(filepath.Join(cache, "ns-1", "bin", "tool")); err != nil || runtime.GOOS != "windows" && fi.Mode().Perm()&0o100 == 0 {
				t.Errorf("executable mode = %v, %v", fi, err)
			}
			if leftovers, _ := filepath.Glob(filepath.Join(cache, ".*")); len(leftovers) != 0 {
				t.Errorf("staging files left: %q", leftovers)
			}
		})
		t.Run("zip without root", func(t *testing.T) {
			t.Parallel()
			archive := zipArchive(t, entry{name: "tool", body: "exe"}, entry{name: "other", body: "x"})
			u, _ := serve(t, "ns-2.zip", archive, digestOf(archive))
			dir := filepath.Join(t.TempDir(), "ns-2")
			exe, err := Install(t.Context(), http.DefaultClient, u, "", dir, "tool")
			if err != nil {
				t.Fatal(err)
			}
			if b, err := os.ReadFile(exe); err != nil || string(b) != "exe" {
				t.Fatalf("exe = %q, %v", b, err)
			}
			if b, err := os.ReadFile(filepath.Join(dir, "other")); err != nil || string(b) != "x" {
				t.Errorf("other = %q, %v", b, err)
			}
		})
		t.Run("replaces an incomplete install", func(t *testing.T) {
			t.Parallel()
			cache := t.TempDir()
			if err := os.MkdirAll(filepath.Join(cache, "ns-3", "bin"), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(cache, "ns-3", ".sha256"), []byte("old"), 0o644); err != nil {
				t.Fatal(err)
			}
			archive := tarGz(t, entry{name: "ns-3/bin/tool", body: "new"})
			u, _ := serve(t, "ns-3.tar.gz", archive, digestOf(archive))
			exe, err := Install(t.Context(), http.DefaultClient, u, "ns-3", filepath.Join(cache, "ns-3"), "bin/tool")
			if err != nil {
				t.Fatal(err)
			}
			if b, err := os.ReadFile(exe); err != nil || string(b) != "new" {
				t.Fatalf("exe = %q, %v", b, err)
			}
		})
	})
	t.Run("error", func(t *testing.T) {
		t.Parallel()
		exe := entry{name: "ns/bin/tool", body: "exe"}
		for _, tc := range []struct {
			name    string
			archive []byte
		}{
			// Each unsafe archive is otherwise complete.
			{"traversal", tarGz(t, exe, entry{name: "../evil", body: "x"})},
			{"absolute", tarGz(t, exe, entry{name: "/evil", body: "x"})},
			{"escaping symlink", tarGz(t, exe, entry{name: "ns/lib/x", link: "../../../evil"})},
			{"no executable", tarGz(t, entry{name: "ns/lib/x", body: "x"})},
			{"hard link", func() []byte {
				var b bytes.Buffer
				gz := gzip.NewWriter(&b)
				tw := tar.NewWriter(gz)
				_ = tw.WriteHeader(&tar.Header{Name: exe.name, Mode: 0o755, Size: 3, Typeflag: tar.TypeReg})
				_, _ = tw.Write([]byte("exe"))
				_ = tw.WriteHeader(&tar.Header{Name: "ns/bin/x", Typeflag: tar.TypeLink, Linkname: "/etc/passwd"})
				_ = tw.Close()
				_ = gz.Close()
				return b.Bytes()
			}()},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				u, _ := serve(t, "ns.tar.gz", tc.archive, digestOf(tc.archive))
				cache := t.TempDir()
				if _, err := Install(t.Context(), http.DefaultClient, u, "ns", filepath.Join(cache, "ns"), "bin/tool"); err == nil {
					t.Fatal("unsafe archive accepted")
				}
				if _, err := os.Stat(filepath.Join(cache, "ns")); !os.IsNotExist(err) {
					t.Errorf("install directory exists: %v", err)
				}
			})
		}
		t.Run("digest mismatch", func(t *testing.T) {
			t.Parallel()
			archive := tarGz(t, entry{name: "ns/bin/tool", body: "tampered"})
			u, _ := serve(t, "ns.tar.gz", archive, digestOf([]byte("expected")))
			cache := t.TempDir()
			if _, err := Install(t.Context(), http.DefaultClient, u, "ns", filepath.Join(cache, "ns"), "bin/tool"); err == nil || !strings.Contains(err.Error(), "SHA-256") {
				t.Fatalf("error = %v", err)
			}
			if entries, _ := os.ReadDir(cache); len(entries) != 0 {
				t.Errorf("cache not empty: %v", entries)
			}
		})
		t.Run("malformed checksum", func(t *testing.T) {
			t.Parallel()
			u, _ := serve(t, "ns.tar.gz", nil, "not-hex")
			if _, err := Install(t.Context(), http.DefaultClient, u, "ns", filepath.Join(t.TempDir(), "ns"), "bin/tool"); err == nil || !strings.Contains(err.Error(), "malformed") {
				t.Fatalf("error = %v", err)
			}
		})
		t.Run("missing build", func(t *testing.T) {
			t.Parallel()
			u, _ := serve(t, "other.tar.gz", nil, "")
			if _, err := Install(t.Context(), http.DefaultClient, strings.Replace(u, "other", "ns", 1), "ns", filepath.Join(t.TempDir(), "ns"), "bin/tool"); err == nil || !strings.Contains(err.Error(), "release has no ns.tar.gz") {
				t.Fatalf("error = %v", err)
			}
		})
	})
}
