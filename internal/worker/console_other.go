// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Console handling for child processes outside Windows.

//go:build !windows

package worker

import "os/exec"

// HideConsole prevents console allocation on Windows; it does nothing here.
func HideConsole(*exec.Cmd) {}
