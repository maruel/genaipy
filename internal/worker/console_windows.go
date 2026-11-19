// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Console handling for child processes on Windows.

package worker

import (
	"os/exec"
	"syscall"
)

// HideConsole runs cmd in a console without a window. Console children, such
// as the Python process uv launches, inherit it instead of opening their own;
// explicit stdout and stderr handles still carry their logs.
func HideConsole(cmd *exec.Cmd) {
	const createNoWindow = 0x08000000
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= createNoWindow
}
