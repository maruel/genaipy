// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package uv installs the pinned uv release that runs the Python workers.
//
// Workers do not depend on a uv on PATH. The pinned binary shares uv's own
// package and Python caches with any other uv on the host.
package uv

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"

	"github.com/maruel/genaipy/internal/release"
)

// Version is the pinned astral-sh/uv release.
const Version = "0.13.8"

// Install returns the uv executable, downloading the release build for this
// platform into the user cache directory unless it is already there.
func Install(ctx context.Context) (string, error) {
	cache, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("get user cache dir: %w", err)
	}
	u, root, exe, err := build()
	if err != nil {
		return "", err
	}
	exe, err = release.Install(ctx, http.DefaultClient, u, root, filepath.Join(cache, "genaipy", "uv", Version), exe)
	if err != nil {
		return "", fmt.Errorf("uv %s: %w", Version, err)
	}
	return exe, nil
}

// build returns the archive URL of this platform's build, the archive
// directory holding the executable, and the executable name. The tar.gz
// archives hold a directory named after the target; the zip archives do not.
func build() (u, root, exe string, err error) {
	arch := map[string]string{"amd64": "x86_64", "arm64": "aarch64"}[runtime.GOARCH]
	target := map[string]string{"darwin": "apple-darwin", "linux": "unknown-linux-gnu", "windows": "pc-windows-msvc"}[runtime.GOOS]
	if arch == "" || target == "" {
		return "", "", "", fmt.Errorf("uv has no build for %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	name := "uv-" + arch + "-" + target
	base := "https://github.com/astral-sh/uv/releases/download/" + Version + "/" + name
	if runtime.GOOS == "windows" {
		return base + ".zip", "", "uv.exe", nil
	}
	return base + ".tar.gz", name, "uv", nil
}
