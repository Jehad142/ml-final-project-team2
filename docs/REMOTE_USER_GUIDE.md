# Team Two Quick Start Guide

Welcome to Team Two's container environment. This guide shows how to configure automatic SSH login (shared secret) and set up GitHub access without entering a password.

---

## SSH: Configure a shared secret for passwordless login

### Step 1 — Generate an SSH key (if you don’t have one)
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press Enter to accept the default location (`~/.ssh/id_ed25519`). Optionally set a passphrase.

### Step 2 — Copy your public key to the server
If the server uses a nonstandard port, include it:
```bash
ssh-copy-id -p 5555 <user-name>@<remote-host>
```
Replace `5555` and `<user-name>@<remote-host>` with the correct port, user, and host.

If `ssh-copy-id` is not available:
```bash
cat ~/.ssh/id_ed25519.pub | ssh -p 5555 <user-name>@<remote-host> 'mkdir -p ~/.ssh && umask 077 && cat >> ~/.ssh/authorized_keys'
```

### Step 3 — Verify passwordless SSH
```bash
ssh -p 5555 <user-name>@<remote-host>
```

### Optional: Use an SSH config entry
Add host aliases to `~/.ssh/config`:
```
Host docker-host
  HostName <remote-host>
  Port 22
  User <user-name>

Host docker-container
  HostName <remote-host>
  Port 5555
  User <user-name>
```
Then connect with:
```bash
ssh docker-host
ssh docker-container
```

---

## GitHub: Configure passwordless Git access via SSH

### Step 1 — Ensure your SSH key is available locally
```bash
cat ~/.ssh/id_ed25519.pub
```

### Step 2 — Add the public key to your GitHub account
1. Go to [GitHub → Settings → SSH and GPG keys](https://github.com/settings/keys).
2. Click New SSH key, paste the contents of `~/.ssh/id_ed25519.pub`, and save.

### Step 3 — Use SSH URLs for repositories
Instead of:
```bash
https://github.com/team2/ml-final-project-team2.git
```
Use:
```bash
git@github.com:team2/ml-final-project-team2.git
```

### Step 4 — Make Git prefer SSH over HTTPS (optional)
```bash
git config --global url."git@github.com:".insteadOf "https://github.com/"
```

### Step 5 — Verify GitHub SSH authentication
```bash
ssh -T git@github.com
```
Expected output:
```
Hi your-username! You've successfully authenticated, but GitHub does not provide shell access.
```

---

## Helpful notes and troubleshooting

- Specify a key for GitHub in `~/.ssh/config`:
  ```
  Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
  ```
- If using a passphrase, add your key to the agent:
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  ```
- Remove a known host for a host:port pair:
  ```bash
  ssh-keygen -R [<remote-host>]:5555
  ```

---

## Security recommendations

- Use `ed25519` keys or RSA with 3072+ bits.
- Protect keys with passphrases and use an SSH agent.
- Rotate keys periodically and revoke lost keys.
- Prefer per-user keys over shared private keys.

