// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Console handling for child processes outside Windows.

//go:build !windows

package nemospeechsrv

import "os/exec"

func hideConsole(*exec.Cmd) {}
