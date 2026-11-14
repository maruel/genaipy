// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Console handling for child processes on Windows.

package nemospeechsrv

import (
	"os/exec"
	"syscall"
)

// hideConsole prevents console allocation; explicit stdout and stderr handles
// still carry server logs.
func hideConsole(cmd *exec.Cmd) {
	const createNoWindow = 0x08000000
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= createNoWindow
}
