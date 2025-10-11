"""
Remote command execution utilities.
"""
import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class CommandResult:
    """Result of a remote command execution"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int

class RemoteExecutor:
    """Execute commands on remote host"""
    def __init__(self, host: str):
        self.host = host
        
    def _build_ssh_command(self, command: str, use_sudo: bool = False) -> List[str]:
        """Build the SSH command array"""
        # Base command becomes: ssh host 'sudo -n command' or ssh host 'command'
        remote_cmd = f"sudo -n {command}" if use_sudo else command
        return ["ssh", self.host, remote_cmd]  # ssh handles the quoting
        
    def run_command(self, command: str, use_sudo: bool = False, timeout: int = 30) -> CommandResult:
        """Run single command on remote host
        
        Args:
            command: The command to execute
            use_sudo: Whether to use sudo -n prefix
            timeout: Command timeout in seconds
            
        Returns:
            CommandResult with execution details
        """
        ssh_cmd = self._build_ssh_command(command, use_sudo)
        
        try:
            # Run command with timeout
            logger.info(f"Executing: {' '.join(ssh_cmd)}")
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                check=False,  # Don't raise on error
                timeout=timeout  # Add timeout
            )
            
            success = result.returncode == 0
            if not success:
                logger.error(f"Command failed with exit code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
            
            return CommandResult(
                success=success,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode
            )
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout} seconds: {e}")
            return CommandResult(
                success=False,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                exit_code=-1
            )
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to execute command: {e}")
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                exit_code=-1
            )
        
    def run_commands(self, commands: List[str], use_sudo: bool = False) -> List[CommandResult]:
        """Run multiple commands on remote host
        
        Args:
            commands: List of commands to execute
            use_sudo: Whether to use sudo for all commands
            
        Returns:
            List of CommandResults
        """
        return [self.run_command(cmd, use_sudo) for cmd in commands]