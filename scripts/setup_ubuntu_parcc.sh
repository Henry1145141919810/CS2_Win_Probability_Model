#!/usr/bin/env bash
# Configure a fresh Ubuntu WSL distribution for the PARCC Betty cluster.
# Runs as root inside WSL. Called by scripts/setup_wsl_parcc.ps1; can be re-run safely.
#
#   bash setup_ubuntu_parcc.sh <linux-username>     (username = your PennKey, e.g. hyhuang)
#
# What it does (per https://parcc.upenn.edu/training/getting-started/logging-in/):
#   * installs krb5-user (kinit/klist), openssh-client, rsync, git, tmux, python3, pipx
#   * writes /etc/krb5.conf for the UPENN.EDU realm (KDCs verified via DNS SRV records)
#   * creates the user, passwordless sudo, sets it as the WSL default user, enables systemd
#   * writes ~/.ssh/config with the PARCC-recommended block (GSSAPI, VerifyHostKeyDNS, multiplexing)
#   * generates an ed25519 key (register it on Betty with ssh-copy-id to skip Duo: key + kinit = 2 of 3)
#   * installs globus-cli for large transfers (login nodes must not be used for heavy rsync/scp)
set -euo pipefail

USER_NAME="${1:-hyhuang}"
REALM="UPENN.EDU"
export DEBIAN_FRONTEND=noninteractive

echo "== apt packages =="
# preseed krb5-config so apt never prompts for the realm
debconf-set-selections <<EOF
krb5-config krb5-config/default_realm string ${REALM}
krb5-config krb5-config/kerberos_servers string kerberos1.upenn.edu kerberos2.upenn.edu kerberos3.upenn.edu kerberos4.upenn.edu
krb5-config krb5-config/admin_server string kerberos1.upenn.edu
krb5-config krb5-config/add_servers boolean false
krb5-config krb5-config/add_servers_realm string ${REALM}
EOF
apt-get update -qq
apt-get install -y -qq krb5-user openssh-client rsync git tmux curl wget unzip ca-certificates \
    python3 python3-venv python3-pip pipx build-essential dos2unix >/dev/null

echo "== /etc/krb5.conf =="
cat > /etc/krb5.conf <<EOF
[libdefaults]
    default_realm = ${REALM}
    dns_lookup_kdc = true
    dns_lookup_realm = true
    ticket_lifetime = 10h
    renew_lifetime = 7d
    forwardable = true
    rdns = false

[realms]
    ${REALM} = {
        kdc = kerberos1.upenn.edu
        kdc = kerberos2.upenn.edu
        kdc = kerberos3.upenn.edu
        kdc = kerberos4.upenn.edu
        admin_server = kerberos1.upenn.edu
    }

[domain_realm]
    .upenn.edu = ${REALM}
    upenn.edu = ${REALM}
EOF

echo "== user ${USER_NAME} =="
if ! id -u "${USER_NAME}" >/dev/null 2>&1; then
    useradd -m -s /bin/bash -G sudo "${USER_NAME}"
fi
echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/90-wsl-${USER_NAME}"
chmod 440 "/etc/sudoers.d/90-wsl-${USER_NAME}"

cat > /etc/wsl.conf <<EOF
[user]
default=${USER_NAME}

[boot]
systemd=true

[interop]
appendWindowsPath=true
EOF

HOME_DIR="$(getent passwd "${USER_NAME}" | cut -d: -f6)"
echo "== ssh config in ${HOME_DIR}/.ssh =="
install -d -m 700 -o "${USER_NAME}" -g "${USER_NAME}" "${HOME_DIR}/.ssh"
cat > "${HOME_DIR}/.ssh/config" <<EOF
# PARCC Betty (https://parcc.upenn.edu/training/getting-started/logging-in/)
Host betty
    HostName login.betty.parcc.upenn.edu

Host *.parcc.upenn.edu betty
    User ${USER_NAME}
    VerifyHostKeyDNS yes
    GSSAPIAuthentication yes
    GSSAPIDelegateCredentials no
    ControlMaster auto
    ControlPath ~/.ssh/control:%h:%p:%r
    ControlPersist 10m
    ServerAliveInterval 60
EOF
chown "${USER_NAME}:${USER_NAME}" "${HOME_DIR}/.ssh/config"
chmod 600 "${HOME_DIR}/.ssh/config"

# ed25519 key: once registered on Betty (ssh-copy-id), key + Kerberos ticket satisfies 2-of-3 without Duo
if [ ! -f "${HOME_DIR}/.ssh/id_ed25519" ]; then
    su - "${USER_NAME}" -c 'ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C "$(whoami)@wsl-parcc"'
fi

echo "== shell helpers =="
cat > "${HOME_DIR}/.bash_aliases" <<EOF
# ---- PARCC Betty helpers (see docs/PARCC_betty_guide.md) ----
alias kb='kinit ${USER_NAME}@${REALM}'      # 10 h Kerberos ticket (Duo follows on first ssh unless a key is registered)
alias kl='klist'
alias betty='ssh betty'
alias bettyproj='ssh -t betty "cd /vast/projects/ajw/wharton/cs2-rwp && exec bash -l"'
alias bettymux='ssh -fNM betty'              # persistent master connection; later ssh/scp reuse it
EOF
chown "${USER_NAME}:${USER_NAME}" "${HOME_DIR}/.bash_aliases"

echo "== globus-cli (pipx) =="
su - "${USER_NAME}" -c 'pipx ensurepath >/dev/null 2>&1; pipx install globus-cli >/dev/null 2>&1 || pipx upgrade globus-cli >/dev/null 2>&1 || true'

echo "== verify =="
kinit --version 2>&1 | head -1 || true
klist -V 2>&1 | head -1 || true
ssh -V 2>&1
echo "default user: $(grep -A1 '^\[user\]' /etc/wsl.conf | tail -1)"
echo "public key (register on Betty with: ssh-copy-id betty):"
cat "${HOME_DIR}/.ssh/id_ed25519.pub"
echo "DONE. Restart the distro (wsl --terminate <distro>) so the default user and systemd take effect."
