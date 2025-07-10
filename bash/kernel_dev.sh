#!/bin/bash

# Kernel Development Environment Script
# Usage: ./kernel_dev.sh [option] [arguments]

# Colors for terminal output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

# Determine base directory
BASE_DIR=""
if [ "$(basename "$PWD")" == "kernel_dev_env" ]; then
    BASE_DIR="$PWD"
else
    BASE_DIR="$PWD/kernel_dev_env"
fi

# Default architecture
ARCH="${ARCH:-x86_64}"

# Function: Get kernel download URL based on version
get_kernel_url() {
    local VERSION="$1"
    local MAJOR_VERSION=$(echo "$VERSION" | cut -d. -f1)
    
    # Determine download base URL based on major version
    if [ "$MAJOR_VERSION" -eq 2 ]; then
        BASE_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/v2.x"
    elif [ "$MAJOR_VERSION" -eq 3 ]; then
        BASE_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/v3.x"
    elif [ "$MAJOR_VERSION" -eq 4 ]; then
        BASE_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/v4.x"
    elif [ "$MAJOR_VERSION" -eq 5 ]; then
        BASE_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/v5.x"
    elif [ "$MAJOR_VERSION" -eq 6 ]; then
        BASE_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/v6.x"
    else
        # Default to v5.x or higher
        BASE_URL="https://mirrors.edge.kernel.org/pub/linux/kernel/v5.x"
    fi
    
    echo "$BASE_URL/linux-$VERSION.tar.xz"
}

# Function: Download kernel
download_kernel() {
    local KERNEL_VERSION="${1:-5.15.0}"
    local KERNEL_ARCHIVE="linux-$KERNEL_VERSION.tar.xz"
    local KERNEL_URL=$(get_kernel_url "$KERNEL_VERSION")
    local KERNEL_SRC_DIR="$BASE_DIR/src/kernel"
    
    echo -e "${YELLOW}Downloading Linux kernel version $KERNEL_VERSION for $ARCH...${NC}"
    
    # Check if wget or curl is installed
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: wget or curl is required to download the kernel.${NC}"
        return 1
    fi
    
    # Create source directory if it doesn't exist
    mkdir -p "$BASE_DIR/src"
    
    cd "$BASE_DIR/src" || { echo -e "${RED}Failed to access source directory!${NC}"; return 1; }
    
    # Download kernel archive
    if [ -f "$KERNEL_ARCHIVE" ]; then
        echo -e "${YELLOW}Kernel archive already exists, skipping download.${NC}"
    else
        if command -v wget &> /dev/null; then
            wget -c "$KERNEL_URL"
        else
            curl -L -O "$KERNEL_URL"
        fi
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download kernel archive!${NC}"
            return 1
        fi
    fi
    
    # Extract kernel source
    if [ -d "$KERNEL_SRC_DIR" ]; then
        echo -e "${YELLOW}Kernel source directory exists, removing old version...${NC}"
        rm -rf "$KERNEL_SRC_DIR"
    fi
    
    echo -e "${YELLOW}Extracting kernel source...${NC}"
    tar -xJf "$KERNEL_ARCHIVE"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to extract kernel source!${NC}"
        return 1
    fi
    
    # Rename extracted directory to 'kernel'
    mv "linux-$KERNEL_VERSION" "$KERNEL_SRC_DIR"
    
    echo -e "${GREEN}Kernel version $KERNEL_VERSION downloaded and prepared successfully!${NC}"
    echo -e "${GREEN}Kernel source located at: $KERNEL_SRC_DIR${NC}"
    return 0
}

# Function: Verify kernel source exists
verify_kernel_source() {
    local KERNEL_SRC_DIR="$BASE_DIR/src/kernel"
    if [ ! -d "$KERNEL_SRC_DIR" ]; then
        echo -e "${RED}Kernel source not found! Please download it first.${NC}"
        return 1
    fi
    return 0
}

# Function: Build kernel
build_kernel() {
    verify_kernel_source || return 1
    
    echo -e "${YELLOW}Building kernel for $ARCH...${NC}"
    cd "$BASE_DIR/src/kernel" || { echo -e "${RED}Failed to access kernel source!${NC}"; return 1; }
    
    # Create build directory if it doesn't exist
    mkdir -p "$BASE_DIR/build"
    
    # Clean previous build
    make clean
    
    # Set architecture
    case "$ARCH" in
        x86)
            ARCH_PARAM="i386"
            ;;
        x86_64)
            ARCH_PARAM="x86_64"
            ;;
        arm32)
            ARCH_PARAM="arm"
            ;;
        aarch64)
            ARCH_PARAM="arm64"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Configure kernel with necessary features
    if [ ! -f ".config" ]; then
        echo -e "${YELLOW}No kernel config found, using default for $ARCH...${NC}"
        make "$ARCH_PARAM"_defconfig
        # Enable 9P filesystem support for folder sharing
        echo "CONFIG_9P_FS=y" >> .config
        echo "CONFIG_9P_FS_POSIX_ACL=y" >> .config
        echo "CONFIG_NET_9P=y" >> .config
        echo "CONFIG_NET_9P_VIRTIO=y" >> .config
        # Ensure configuration is consistent
        make olddefconfig ARCH="$ARCH_PARAM"
    fi
    
    # Build kernel
    make -j$(nproc) ARCH="$ARCH_PARAM"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Kernel build failed!${NC}"
        return 1
    fi
    
    # Determine kernel image path based on architecture
    case "$ARCH" in
        x86|x86_64)
            KERNEL_IMAGE="arch/x86/boot/bzImage"
            ;;
        arm32)
            KERNEL_IMAGE="arch/arm/boot/zImage"
            ;;
        aarch64)
            KERNEL_IMAGE="arch/arm64/boot/Image"
            ;;
    esac
    
    # Copy kernel image to build directory
    cp "$KERNEL_IMAGE" "$BASE_DIR/build/kernel-$ARCH"
    echo -e "${GREEN}Kernel for $ARCH built successfully!${NC}"
    echo -e "${GREEN}Kernel image located at: $BASE_DIR/build/kernel-$ARCH${NC}"
    return 0
}

# Function: Rebuild kernel
rebuild_kernel() {
    verify_kernel_source || return 1
    
    echo -e "${YELLOW}Rebuilding kernel for $ARCH...${NC}"
    cd "$BASE_DIR/src/kernel" || { echo -e "${RED}Failed to access kernel source!${NC}"; return 1; }
    
    # Create build directory if it doesn't exist
    mkdir -p "$BASE_DIR/build"
    
    # Set architecture
    case "$ARCH" in
        x86)
            ARCH_PARAM="i386"
            ;;
        x86_64)
            ARCH_PARAM="x86_64"
            ;;
        arm32)
            ARCH_PARAM="arm"
            ;;
        aarch64)
            ARCH_PARAM="arm64"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Clean previous build
    make clean ARCH="$ARCH_PARAM"
    
    # Assume we have a config file; adjust as needed
    if [ ! -f ".config" ]; then
        echo -e "${YELLOW}No kernel config found, using default for $ARCH...${NC}"
        make "$ARCH_PARAM"_defconfig
    fi
    
    # Rebuild kernel
    make -j$(nproc) ARCH="$ARCH_PARAM"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Kernel rebuild failed!${NC}"
        return 1
    fi
    
    # Determine kernel image path based on architecture
    case "$ARCH" in
        x86|x86_64)
            KERNEL_IMAGE="arch/x86/boot/bzImage"
            ;;
        arm32)
            KERNEL_IMAGE="arch/arm/boot/zImage"
            ;;
        aarch64)
            KERNEL_IMAGE="arch/arm64/boot/Image"
            ;;
    esac
    
    # Copy kernel image to build directory
    cp "$KERNEL_IMAGE" "$BASE_DIR/build/kernel-$ARCH"
    echo -e "${GREEN}Kernel for $ARCH rebuilt successfully!${NC}"
    return 0
}

# Function: Create rootfs with debootstrap
make_rootfs_debootstrap() {
    local DISTRO="${2:-bullseye}"
    echo -e "${YELLOW}Creating rootfs with debootstrap for $ARCH...${NC}"
    
    # Create rootfs directory if it doesn't exist
    mkdir -p "$BASE_DIR/rootfs-debootstrap"
    ROOTFS_DIR="$BASE_DIR/rootfs-debootstrap"
    
    # Check if debootstrap is installed
    if ! command -v debootstrap &> /dev/null; then
        echo -e "${RED}Error: debootstrap is required for this method.${NC}"
        return 1
    fi
    
    # Determine debootstrap architecture
    case "$ARCH" in
        x86)
            DEBOOTSTRAP_ARCH="i386"
            ;;
        x86_64)
            DEBOOTSTRAP_ARCH="amd64"
            ;;
        arm32)
            DEBOOTSTRAP_ARCH="armhf"
            ;;
        aarch64)
            DEBOOTSTRAP_ARCH="arm64"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Clean previous rootfs
    rm -rf "$ROOTFS_DIR"/*
    
    # Create basic structure
    mkdir -p "$ROOTFS_DIR"{bin,sbin,etc,lib,lib64,dev,proc,sys,home,root,tmp,var}
    
    # Run debootstrap
    echo -e "${YELLOW}Running debootstrap for $DISTRO...${NC}"
    debootstrap --arch="$DEBOOTSTRAP_ARCH" "$DISTRO" "$ROOTFS_DIR"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}debootstrap failed!${NC}"
        return 1
    fi
    
    # Create necessary system files
    mkdir -p "$ROOTFS_DIR/etc/init.d"
    
    # Create init script
    cat > "$ROOTFS_DIR/init" << EOF
#!/bin/sh
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs none /dev
mkdir -p /dev/pts /dev/shm
mount -t devpts devpts /dev/pts
mount -t tmpfs tmpfs /dev/shm
echo "Mounting root filesystem..."
exec /sbin/init
EOF
    chmod +x "$ROOTFS_DIR/init"
    cat > "$ROOTFS_DIR/etc/inittab" << EOF
::sysinit:/etc/init.d/rcS
::askfirst:-/bin/sh
EOF
    
    cat > "$ROOTFS_DIR/etc/init.d/rcS" << EOF
#!/bin/sh
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev
mkdir -p /dev/pts
mount -t devpts devpts /dev/pts
echo "Starting $ARCH rootfs..."
#exec /sbin/init
EOF
    chmod +x "$ROOTFS_DIR/etc/init.d/rcS"
    
    echo -e "${GREEN}Rootfs created with debootstrap successfully!${NC}"
    echo -e "${GREEN}Rootfs located at: $ROOTFS_DIR${NC}"
    return 0
}
verify_busybox_installation() {
    local rootfs_dir="$1"
    local failed=0

    # Check busybox binary
    if [ ! -x "$rootfs_dir/bin/busybox" ]; then
        echo -e "${RED}Busybox binary not found or not executable${NC}"
        return 1
    fi

    # Test busybox functionality
    "$rootfs_dir/bin/busybox" --help >/dev/null 2>&1 || {
        echo -e "${RED}Busybox binary test failed${NC}"
        failed=1
    }

    # Verify essential symlinks
    for cmd in sh init mount ls cp chmod; do
        if [ ! -L "$rootfs_dir/bin/$cmd" ] || [ ! -e "$rootfs_dir/bin/$cmd" ]; then
            echo -e "${RED}Missing or broken symlink: /bin/$cmd${NC}"
            failed=1
        fi
    done

    # Verify /sbin/init symlink specifically
    if [ ! -L "$rootfs_dir/sbin/init" ] || [ ! -e "$rootfs_dir/sbin/init" ]; then
        echo -e "${RED}Missing or broken symlink: /sbin/init${NC}"
        failed=1
    fi

    return $failed
}
# # Verify busybox installation
# verify_busybox_installation() {
#     local rootfs_dir="$1"
#     local failed=0

#     # Check busybox binary
#     if [ ! -x "$rootfs_dir/bin/busybox" ]; then
#         echo -e "${RED}Busybox binary is not executable${NC}"
#         return 1
#     fi

#     # Check essential symlinks in /bin
#     for cmd in sh init mount ls cp chmod; do
#         if [ ! -L "$rootfs_dir/bin/$cmd" ] || [ ! -e "$rootfs_dir/bin/$cmd" ]; then
#             echo -e "${RED}Missing or broken symlink in /bin: $cmd${NC}"
#             failed=1
#         fi
#     done

#     # Check /sbin/init symlink specifically
#     if [ ! -L "$rootfs_dir/sbin/init" ] || [ ! -e "$rootfs_dir/sbin/init" ]; then
#         echo -e "${RED}Missing or broken symlink: /sbin/init${NC}"
#         failed=1
#     fi

#     # Verify /sbin/init points to busybox
#     if [ "$(readlink "$rootfs_dir/sbin/init")" != "../bin/busybox" ]; then
#         echo -e "${RED}/sbin/init symlink points to wrong target${NC}"
#         failed=1
#     fi

#     # Test basic busybox functionality
#     if ! "$rootfs_dir/bin/busybox" --help >/dev/null 2>&1; then
#         echo -e "${RED}Basic busybox functionality test failed${NC}"
#         failed=1
#     fi

#     return $failed
# }

# Function: Verify rootfs structure
verify_rootfs_structure() {
    local ROOTFS_DIR="$1"
    local failed=0

    # Check if rootfs directory exists
    if [ ! -d "$ROOTFS_DIR" ]; then
        echo -e "${RED}Rootfs directory not found: $ROOTFS_DIR${NC}"
        return 1
    fi

    # Check essential directories
    local essential_dirs=("bin" "sbin" "etc" "proc" "sys" "dev" "tmp" "root" "var")
    for dir in "${essential_dirs[@]}"; do
        if [ ! -d "$ROOTFS_DIR/$dir" ]; then
            echo -e "${RED}Missing essential directory: $dir${NC}"
            failed=1
        fi
    done

    # Check essential files
    if [ ! -f "$ROOTFS_DIR/init" ]; then
        echo -e "${RED}Missing init script${NC}"
        failed=1
    fi

    if [ ! -f "$ROOTFS_DIR/etc/inittab" ]; then
        echo -e "${RED}Missing /etc/inittab${NC}"
        failed=1
    fi

    if [ ! -f "$ROOTFS_DIR/etc/init.d/rcS" ]; then
        echo -e "${RED}Missing /etc/init.d/rcS${NC}"
        failed=1
    fi

    # Check essential symlinks
    local essential_symlinks=("/bin/getty" "/sbin/getty" "/bin/init" "/sbin/init")
    for symlink in "${essential_symlinks[@]}"; do
        if [ ! -L "$ROOTFS_DIR$symlink" ]; then
            echo -e "${RED}Missing symlink: $symlink${NC}"
            failed=1
        fi
    done

    return $failed
}

# Function: Create rootfs with busybox
make_rootfs_busybox() {
    local BUSYBOX_VERSION="${2:-1.35.0}"
    echo -e "${YELLOW}Creating rootfs with busybox for $ARCH...${NC}"
    
    # Create rootfs directory if it doesn't exist
    mkdir -p "$BASE_DIR/rootfs-busybox"
    ROOTFS_DIR="$BASE_DIR/rootfs-busybox"
    
    # Download busybox if not already downloaded
    build_busybox
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download busybox!${NC}"
        return 1
    fi
    
    # Clean previous rootfs
    rm -rf "$ROOTFS_DIR"/*
    
    # Create basic rootfs structure
    for dir in bin sbin etc lib lib64 dev proc sys home root tmp var; do
        mkdir -p "$ROOTFS_DIR/$dir"
    done
    
    # Install busybox
    BUSYBOX_BINARY="$BASE_DIR/bin/busybox/busybox"
    if [ ! -f "$BUSYBOX_BINARY" ]; then
        echo -e "${RED}Busybox binary not found at $BUSYBOX_BINARY${NC}"
        return 1
    fi
    
    # Copy and set permissions with error checking
    echo -e "${YELLOW}Copying busybox binary...${NC}"
    if ! cp "$BUSYBOX_BINARY" "$ROOTFS_DIR/bin/busybox"; then
        echo -e "${RED}Failed to copy busybox binary${NC}"
        return 1
    fi
    
    if ! chmod 755 "$ROOTFS_DIR/bin/busybox"; then
        echo -e "${RED}Failed to set busybox permissions${NC}"
        return 1
    fi
    
    # Create symlinks for busybox commands
    echo -e "${YELLOW}Creating busybox symlinks...${NC}"
    if ! cd "$ROOTFS_DIR/bin"; then
        echo -e "${RED}Failed to access bin directory${NC}"
        return 1
    fi
    
    # Create essential symlinks first
    for cmd in sh init getty; do
        if ! ln -sf busybox "$cmd"; then
            echo -e "${RED}Failed to create symlink for $cmd${NC}"
            return 1
        fi
    done

    # Create getty in sbin
    cd "$ROOTFS_DIR/sbin" || {
        echo -e "${RED}Failed to access sbin directory${NC}"
        return 1
    }
    if ! ln -sf ../bin/busybox getty; then
        echo -e "${RED}Failed to create /sbin/getty symlink${NC}"
        return 1
    fi
    cd "$ROOTFS_DIR/bin" || {
        echo -e "${RED}Failed to return to bin directory${NC}"
        return 1
    }

    # Create /sbin/init symlink
    cd "$ROOTFS_DIR/sbin" || {
        echo -e "${RED}Failed to access sbin directory${NC}"
        return 1
    }
    if ! ln -sf ../bin/busybox init; then
        echo -e "${RED}Failed to create /sbin/init symlink${NC}"
        return 1
    fi
    cd "$ROOTFS_DIR/bin" || {
        echo -e "${RED}Failed to return to bin directory${NC}"
        return 1
    }
    
    # Create symlinks for all busybox commands
    if ! "$ROOTFS_DIR/bin/busybox" --list > /dev/null 2>&1; then
        echo -e "${RED}Failed to list busybox commands${NC}"
        return 1
    fi
    
    "$ROOTFS_DIR/bin/busybox" --list | while read -r cmd; do
        if ! ln -sf busybox "$cmd"; then
            echo -e "${RED}Failed to create symlink for $cmd${NC}"
            continue
        fi
    done


    # Run busybox verification
    echo -e "${YELLOW}Verifying busybox installation...${NC}"
    if ! verify_busybox_installation "$ROOTFS_DIR"; then
        echo -e "${RED}Busybox installation verification failed!${NC}"
        return 1
    fi
    echo -e "${GREEN}Busybox installation verified successfully!${NC}"
    
    # Create necessary files
    # Create essential system configuration files
    echo "localhost" > "$ROOTFS_DIR/etc/hostname"

    cat > "$ROOTFS_DIR/etc/fstab" << EOF
proc            /proc           proc        defaults            0 0
sysfs           /sys            sysfs       defaults            0 0
devtmpfs        /dev            devtmpfs    mode=0755,nosuid    0 0
tmpfs           /dev/shm        tmpfs       mode=1777,nosuid    0 0
devpts          /dev/pts        devpts      mode=0620,gid=5     0 0
EOF

    # Create minimal passwd and group files
    cat > "$ROOTFS_DIR/etc/passwd" << EOF
root:x:0:0:root:/root:/bin/sh
EOF
    # Set root to no password
    echo 'root::0:0:99999:7:::' > $ROOTFS_DIR/etc/shadow
    chmod 600 $ROOTFS_DIR/etc/shadow

    cat > "$ROOTFS_DIR/etc/group" << EOF
root:x:0:
tty:x:5:
EOF

    # Create /etc/profile for environment setup
    cat > "$ROOTFS_DIR/etc/profile" << EOF
# Set up environment variables
PATH=/bin:/sbin:/usr/bin:/usr/sbin
TERM=linux
HOSTNAME=\$(cat /etc/hostname)
PS1='[\u@\h \W]\$ '
export PATH TERM HOSTNAME PS1
EOF
    
    cat > "$ROOTFS_DIR/etc/inittab" << EOF
::sysinit:/etc/init.d/rcS
::respawn:/sbin/getty -L tty1 9600
::respawn:/sbin/getty -L tty2 9600
::respawn:/sbin/getty -L tty3 9600
::respawn:/sbin/getty -L tty4 9600
EOF
    
    mkdir -p "$ROOTFS_DIR/etc/init.d"
    # Create init script
    cat > "$ROOTFS_DIR/init" << EOF
#!/bin/sh

# Mount essential filesystems
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs none /dev

# Create and mount additional filesystems
mkdir -p /dev/pts /dev/shm
mount -t devpts none /dev/pts
mount -t tmpfs none /dev/shm

echo "Starting $ARCH rootfs..."

# Ensure /sbin exists and busybox init is available
mkdir -p /sbin
ln -sf /bin/busybox /sbin/init

# Start system initialization
exec /sbin/init
EOF
    chmod 755 "$ROOTFS_DIR/init"

    # Create rcS script
    cat > "$ROOTFS_DIR/etc/init.d/rcS" << EOF
#!/bin/sh

# System initialization
echo "Initializing system..."

# Set up hostname
hostname localhost

# Start basic services
/bin/mount -a
/bin/hostname -F /etc/hostname

echo "System initialization complete."
EOF
    chmod 755 "$ROOTFS_DIR/etc/init.d/rcS"
    
    if verify_rootfs_structure "$ROOTFS_DIR" && verify_busybox_installation "$ROOTFS_DIR"; then
        echo -e "${GREEN}Rootfs created with busybox successfully!${NC}"
        echo -e "${GREEN}Rootfs located at: $ROOTFS_DIR${NC}"
        return 0
    else
        echo -e "${RED}Rootfs verification failed!${NC}"
        return 1
    fi
}

# Function: Download pre-built busybox binary
build_busybox() {
    local VERSION="1.21.1"
    local BUSYBOX_URL="https://busybox.net/downloads/binaries/$VERSION"
    local BUSYBOX_BIN_DIR="$BASE_DIR/bin/busybox"
    
    echo -e "${YELLOW}Downloading pre-built busybox version $VERSION...${NC}"
    
    # Create binary directory if it doesn't exist
    mkdir -p "$BUSYBOX_BIN_DIR"
    
    cd "$BUSYBOX_BIN_DIR" || { echo -e "${RED}Failed to access binary directory!${NC}"; return 1; }
    
    # Determine binary name based on architecture
    local BINARY_NAME
    case "$ARCH" in
        x86)
            BINARY_NAME="busybox-i686"
            ;;
        x86_64)
            BINARY_NAME="busybox-x86_64"
            ;;
        arm32)
            BINARY_NAME="busybox-armv6l"
            ;;
        aarch64)
            echo -e "${RED}Pre-built binary not available for aarch64. Please use compilation method instead.${NC}"
            return 1
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Download busybox binary
    if [ ! -f "$BINARY_NAME" ]; then
        if command -v wget &> /dev/null; then
            wget -c "$BUSYBOX_URL/$BINARY_NAME"
        else
            curl -L -o "$BINARY_NAME" "$BUSYBOX_URL/$BINARY_NAME"
        fi
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download busybox binary!${NC}"
            return 1
        fi
    fi
    
    # Make binary executable
    chmod +x "$BINARY_NAME"
    
    # Create symlink to standard name
    ln -sf "$BINARY_NAME" busybox
    
    echo -e "${GREEN}Successfully downloaded pre-built busybox binary!${NC}"
    return 0
    # Enable CBS traffic control instead
    sed -i 's/# CONFIG_TC_CBS is not set/CONFIG_TC_CBS=y/' .config
    
    # Verify and adjust busybox configuration
    verify_busybox_config() {
        local config_file="$1"
        
        # Check if config file exists
        if [ ! -f "$config_file" ]; then
            echo -e "${RED}Config file not found: $config_file${NC}"
            return 1
        fi
        
        # Essential features check
        local required_configs=(
            "CONFIG_STATIC=y"
            "# CONFIG_TC_CBQ is not set"
            "CONFIG_TC_CBS=y"
            "CONFIG_FEATURE_MOUNT_HELPERS=y"
            "CONFIG_FEATURE_MOUNT_VERBOSE=y"
        )
        
        for config in "${required_configs[@]}"; do
            if ! grep -q "^$config" "$config_file"; then
                echo -e "${YELLOW}Adjusting config: $config${NC}"
                if [[ "$config" == *"=y"* ]]; then
                    sed -i "s/^.*${config%=*}.*$/$config/" "$config_file"
                else
                    echo "$config" >> "$config_file"
                fi
            fi
        done
        
        return 0
    }
    
    # Enable static linking
    sed -i 's/^.*CONFIG_STATIC.*$/CONFIG_STATIC=y/' .config
    
    # Verify and adjust configuration
    verify_busybox_config ".config" || return 1
    
    # Build with architecture and static linking
    build_log="$BASE_DIR/build/busybox_build.log"
    mkdir -p "$(dirname "$build_log")"
    
    echo -e "${YELLOW}Building busybox with configuration...${NC}"
    case "$ARCH" in
        x86)
            make -j$(nproc) ARCH=i386 CROSS_COMPILE= CONFIG_STATIC=y V=1 2>&1 | tee "$build_log"
            ;;
        x86_64)
            make -j$(nproc) ARCH=x86_64 CROSS_COMPILE= CONFIG_STATIC=y V=1 2>&1 | tee "$build_log"
            ;;
        arm32)
            make -j$(nproc) ARCH=arm CROSS_COMPILE=arm-linux-gnueabi- CONFIG_STATIC=y V=1 2>&1 | tee "$build_log"
            ;;
        aarch64)
            make -j$(nproc) ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- CONFIG_STATIC=y V=1 2>&1 | tee "$build_log"
            ;;
    esac
    
    # Check build status
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo -e "${RED}Busybox build failed! Check build log at: $build_log${NC}"
        return 1
    fi
    
    # Verify binary
    if [ ! -f "busybox" ]; then
        echo -e "${RED}Busybox binary not found after build!${NC}"
        return 1
    fi
    
    # Test basic functionality
    if ! ./busybox --help >/dev/null 2>&1; then
        echo -e "${RED}Built busybox binary is not functional!${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Busybox built successfully!${NC}"
    return 0
}

# Function: Download ready-made rootfs
download_rootfs() {
    local TYPE="${2:-minimal}"
    echo -e "${YELLOW}Downloading ready-made rootfs for $ARCH...${NC}"
    
    # Create rootfs directory if it doesn't exist
    mkdir -p "$BASE_DIR/rootfs-download"
    ROOTFS_DIR="$BASE_DIR/rootfs-download"
    
    # Check if wget or curl is installed
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        echo -e "${RED}Error: wget or curl is required for this method.${NC}"
        return 1
    fi
    
    # Determine download URL based on architecture and type
    case "$ARCH" in
        x86)
            if [ "$TYPE" == "minimal" ]; then
                ROOTFS_URL="https://example.com/rootfs-x86-minimal.tar.gz"
            else
                ROOTFS_URL="https://example.com/rootfs-x86-full.tar.gz"
            fi
            ;;
        x86_64)
            if [ "$TYPE" == "minimal" ]; then
                ROOTFS_URL="https://example.com/rootfs-x86_64-minimal.tar.gz"
            else
                ROOTFS_URL="https://example.com/rootfs-x86_64-full.tar.gz"
            fi
            ;;
        arm32)
            if [ "$TYPE" == "minimal" ]; then
                ROOTFS_URL="https://example.com/rootfs-arm32-minimal.tar.gz"
            else
                ROOTFS_URL="https://example.com/rootfs-arm32-full.tar.gz"
            fi
            ;;
        aarch64)
            if [ "$TYPE" == "minimal" ]; then
                ROOTFS_URL="https://example.com/rootfs-aarch64-minimal.tar.gz"
            else
                ROOTFS_URL="https://example.com/rootfs-aarch64-full.tar.gz"
            fi
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Clean previous rootfs
    rm -rf "$ROOTFS_DIR"/*
    
    # Download rootfs archive
    cd "$BASE_DIR" || { echo -e "${RED}Failed to access base directory!${NC}"; return 1; }
    ARCHIVE_NAME=$(basename "$ROOTFS_URL")
    
    if [ ! -f "$ARCHIVE_NAME" ]; then
        echo -e "${YELLOW}Downloading $ARCHIVE_NAME...${NC}"
        if command -v wget &> /dev/null; then
            wget -c "$ROOTFS_URL"
        else
            curl -L -O "$ROOTFS_URL"
        fi
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download rootfs!${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}Using existing rootfs archive: $ARCHIVE_NAME${NC}"
    fi
    
    # Extract rootfs
    echo -e "${YELLOW}Extracting rootfs...${NC}"
    tar -xzf "$ARCHIVE_NAME" -C "$ROOTFS_DIR" --strip-components=1
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to extract rootfs!${NC}"
        return 1
    fi
    
    # Verify rootfs structure
    if [ ! -d "$ROOTFS_DIR/bin" ] || [ ! -d "$ROOTFS_DIR/etc" ]; then
        echo -e "${RED}Invalid rootfs structure!${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Ready-made rootfs downloaded and extracted successfully!${NC}"
    echo -e "${GREEN}Rootfs located at: $ROOTFS_DIR${NC}"
    return 0
}

# Function: Make rootfs with specified method
make_rootfs() {
    if [ -z "$1" ]; then
        echo -e "${RED}Please specify rootfs creation method: debootstrap, busybox, or download${NC}"
        return 1
    fi
    
    METHOD="$1"
    echo -e "${YELLOW}Creating rootfs with method: $METHOD for $ARCH...${NC}"
    
    case "$METHOD" in
        debootstrap)
            if [ $# -ge 2 ]; then
                make_rootfs_debootstrap "$METHOD" "$2"
            else
                make_rootfs_debootstrap "$METHOD"
            fi
            ;;
        busybox)
            if [ $# -ge 2 ]; then
                make_rootfs_busybox "$METHOD" "$2"
            else
                make_rootfs_busybox "$METHOD"
            fi
            ;;
        download)
            if [ $# -ge 2 ]; then
                download_rootfs "$METHOD" "$2"
            else
                download_rootfs "$METHOD"
            fi
            ;;
        *)
            echo -e "${RED}Unknown rootfs method: $METHOD${NC}"
            echo "Available methods: debootstrap, busybox, download"
            return 1
            ;;
    esac
    
    return 0
}

# Function: Build kernel module
build_module() {
    verify_kernel_source || return 1
    
    if [ -z "$1" ]; then
        echo -e "${RED}Please specify module directory!${NC}"
        return 1
    fi
    
    MODULE_DIR="$1"
    echo -e "${YELLOW}Building kernel module in $MODULE_DIR for $ARCH...${NC}"
    
    # Create modules directory if it doesn't exist
#    mkdir -p "$BASE_DIR/modules"
    
    # Check if module directory exists
    if [ ! -d "$MODULE_DIR" ]; then
        echo -e "${RED}Module directory not found!${NC}"
        return 1
    fi
    
    cd "$MODULE_DIR" || return 1
    
    # Assume we have a Makefile; adjust as needed
    if [ ! -f "Makefile" ]; then
        echo -e "${RED}Makefile not found in module directory!${NC}"
        return 1
    fi
    
    # Create build/modules directory if it doesn't exist
    mkdir -p "$BASE_DIR/build/modules"
    
    # Set architecture for module build
    case "$ARCH" in
        x86)
            ARCH_PARAM="i386"
            ;;
        x86_64)
            ARCH_PARAM="x86_64"
            ;;
        arm32)
            ARCH_PARAM="arm"
            ;;
        aarch64)
            ARCH_PARAM="arm64"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Build the module
    make -C "$BASE_DIR/src/kernel" M="$PWD" modules ARCH="$ARCH_PARAM" 
    if [ $? -ne 0 ]; then
        echo -e "${RED}Module build failed!${NC}"
        return 1
    fi
    
    # Copy module to build directory
    MODULE_NAME="$(find . -name '*.ko' 2>/dev/null)"
    if [ -n "$MODULE_NAME" ]; then
        set -x
        find . -name '*.ko' -exec cp {} "$BASE_DIR/build/modules" \;
        set +x
        echo -e "${GREEN}Kernel module $MODULE_NAME built successfully!${NC}"
        make -C "$BASE_DIR/src/kernel" M="$PWD" clean ARCH="$ARCH_PARAM" 
    else
        make -C "$BASE_DIR/src/kernel" M="$PWD" clean ARCH="$ARCH_PARAM" 
        echo -e "${RED}No module found after build!${NC}"
        return 1
    fi
    
    return 0
}

# Function: Install kernel module to rootfs
install_module() {
    if [ -z "$1" ]; then
        echo -e "${RED}Please specify module name!${NC}"
        return 1
    fi
    
    MODULE_NAME="$1"
    echo -e "${YELLOW}Installing kernel module $MODULE_NAME to rootfs...${NC}"
    
    # Check if module exists
    if [ ! -f "$BASE_DIR/build/modules/$MODULE_NAME" ]; then
        echo -e "${RED}Module $MODULE_NAME not found!${NC}"
        return 1
    fi
    
    # Determine rootfs directory based on method
    if [ -d "$BASE_DIR/rootfs-debootstrap" ]; then
        ROOTFS_DIR="$BASE_DIR/rootfs-debootstrap"
    elif [ -d "$BASE_DIR/rootfs-busybox" ]; then
        ROOTFS_DIR="$BASE_DIR/rootfs-busybox"
    elif [ -d "$BASE_DIR/rootfs-download" ]; then
        ROOTFS_DIR="$BASE_DIR/rootfs-download"
    else
        echo -e "${RED}Rootfs not found! Please create it first.${NC}"
        return 1
    fi
    
    # Create modules directory in rootfs if needed
    mkdir -p "$ROOTFS_DIR/lib/modules"
    
    # Copy module to rootfs
    cp "$BASE_DIR/build/modules/$MODULE_NAME" "$ROOTFS_DIR/lib/modules/"
    
    # Create depmod command if needed (example)
    cat > "$ROOTFS_DIR/bin/depmod" << EOF
#!/bin/sh
echo "Creating module dependencies..."
EOF
    chmod +x "$ROOTFS_DIR/bin/depmod"
    
    echo -e "${GREEN}Kernel module $MODULE_NAME installed to rootfs successfully!${NC}"
    return 0
}

# Function: Run kernel
run_kernel() {
    local MEMORY_SIZE="${1:-512M}"
    local SHARE_DIR="${2:-}"

    # Check if 9P filesystem support is enabled in kernel config when sharing directory
    if [ -n "$SHARE_DIR" ]; then
        if ! grep -q "CONFIG_9P_FS=y" "$BASE_DIR/src/kernel/.config"; then
            echo -e "${RED}Error: 9P filesystem support is not enabled in kernel!${NC}"
            echo -e "${YELLOW}Please enable the following kernel configs:${NC}"
            echo -e "${YELLOW}- CONFIG_9P_FS=y         (9P filesystem support)${NC}"
            echo -e "${YELLOW}- CONFIG_9P_FS_POSIX_ACL=y (9P POSIX Access Control Lists)${NC}"
            echo -e "${YELLOW}- CONFIG_NET_9P=y        (9P network protocol support)${NC}"
            echo -e "${YELLOW}- CONFIG_NET_9P_VIRTIO=y  (9P virtio transport)${NC}"
            return 1
        fi
    fi
    
    # Create mount point in rootfs for shared directory
    if [ -n "$SHARE_DIR" ]; then
        if [ ! -d "$SHARE_DIR" ]; then
            echo -e "${RED}Shared directory $SHARE_DIR does not exist!${NC}"
            return 1
        fi
    fi
    
    # Verify kernel exists
    KERNEL_IMAGE="$BASE_DIR/build/kernel-$ARCH"
    if [ ! -f "$KERNEL_IMAGE" ]; then
        echo -e "${RED}Kernel image for $ARCH not found! Please build the kernel first.${NC}"
        return 1
    fi
    
    # Find available rootfs directories
    declare -a ROOTFS_DIRS=()
    for dir in "$BASE_DIR/rootfs-debootstrap" "$BASE_DIR/rootfs-busybox" "$BASE_DIR/rootfs-download"; do
        if [ -d "$dir" ]; then
            ROOTFS_DIRS+=("$dir")
        fi
    done

    # Check if any rootfs exists
    if [ ${#ROOTFS_DIRS[@]} -eq 0 ]; then
        echo -e "${RED}No rootfs found! Please create one first.${NC}"
        return 1
    fi

    # If only one rootfs exists, use it directly
    if [ ${#ROOTFS_DIRS[@]} -eq 1 ]; then
        ROOTFS_DIR="${ROOTFS_DIRS[0]}"
        echo -e "${GREEN}Using rootfs: $(basename "$ROOTFS_DIR")${NC}"
    else
        # If multiple rootfs exist, let user select
        echo -e "${YELLOW}Multiple rootfs found. Please select one:${NC}"
        select ROOTFS_DIR in "${ROOTFS_DIRS[@]}"; do
            if [ -n "$ROOTFS_DIR" ]; then
                echo -e "${GREEN}Selected rootfs: $(basename "$ROOTFS_DIR")${NC}"
                break
            else
                echo -e "${RED}Invalid selection. Please try again.${NC}"
            fi
        done
    fi
    # Create mount point in rootfs
    mkdir -p "$ROOTFS_DIR/mnt/host"
    echo -e "${RED}To mount shared directory in guest OS, run:${NC}"
    echo -e "${RED}mount -t 9p -o trans=virtio,version=9p2000.L hostshare /mnt/host${NC}"
    
    echo -e "${YELLOW}Running kernel for $ARCH with ${MEMORY_SIZE} memory...${NC}"
    
    # Create rootfs image (example using cpio)
    cd "$ROOTFS_DIR" || return 1
    
    # Temporarily copy kernel modules if they exist
    if [ -d "$BASE_DIR/build/modules" ] && [ -n "$(ls -A "$BASE_DIR/build/modules")" ]; then
        echo -e "${YELLOW}Copying kernel modules to rootfs...${NC}"
        mkdir -p lib/modules
        cp "$BASE_DIR/build/modules"/* lib/modules/
    fi
    
    # Create cpio archive
    find . | cpio -o -H newc > "$BASE_DIR/build/rootfs-$ARCH.cpio.gz"
    
    # Clean up temporary module copies
    if [ -d "lib/modules" ]; then
        rm -rf lib/modules/*
    fi
    
    # Determine QEMU machine type based on architecture
    case "$ARCH" in
        x86|x86_64)
            MACHINE_TYPE="q35"
            KERNEL_PARAM="bzImage"
            ;;
        arm32)
            MACHINE_TYPE="virt"
            KERNEL_PARAM="zImage"
            ;;
        aarch64)
            MACHINE_TYPE="virt"
            KERNEL_PARAM="Image"
            ;;
        *)
            echo -e "${RED}Unsupported architecture for QEMU: $ARCH${NC}"
            return 1
            ;;
    esac
    
    # Use QEMU to run the kernel (example)
    echo -e "${YELLOW}Starting QEMU with $MACHINE_TYPE machine...${NC}"
    set -x
    QEMU_ARGS=(
        -machine "$MACHINE_TYPE"
        -kernel "$KERNEL_IMAGE"
        -initrd "$BASE_DIR/build/rootfs-$ARCH.cpio.gz"
        -m "$MEMORY_SIZE"
        -serial stdio
        -append "console=ttyS0"
        #-append "console=tty0"
    )

    # Add virtio-9p device if share directory is specified
    if [ -n "$SHARE_DIR" ]; then
        QEMU_ARGS+=(
            -device "virtio-9p-pci,fsdev=host_share,mount_tag=hostshare"
            -fsdev "local,security_model=mapped-xattr,path=$SHARE_DIR,id=host_share"
        )
    fi

    qemu-system-"$ARCH" "${QEMU_ARGS[@]}"
    set +x
    
    echo -e "${GREEN}Kernel execution completed!${NC}"
    return 0
}

# Function: Set architecture
set_architecture() {
    if [ -z "$1" ]; then
        echo -e "${RED}Please specify architecture: x86, x86_64, arm32, or aarch64${NC}"
        return 1
    fi
    
    case "$1" in
        x86|x86_64|arm32|aarch64)
            ARCH="$1"
            echo -e "${GREEN}Architecture set to: $ARCH${NC}"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $1${NC}"
            echo "Available architectures: x86, x86_64, arm32, aarch64"
            return 1
            ;;
    esac
    
    return 0
}

# Function: Initialize complete kernel development environment
init_environment() {
    if [ -z "$1" ]; then
        echo -e "${RED}Please specify architecture: x86, x86_64, arm32, or aarch64${NC}"
        return 1
    fi
    
    # Set target architecture
    set_architecture "$1" || return 1
    
    local KERNEL_VERSION="${2:-5.15.0}"
    local ROOTFS_METHOD="${3:-busybox}"
    local ROOTFS_TYPE="${4:-minimal}"
    
    echo -e "${YELLOW}Initializing kernel development environment for $ARCH...${NC}"
    echo -e "${YELLOW}-----------------------------------------------${NC}"
    
    # Step 1: Download kernel
    echo -e "${YELLOW}[1/4] Downloading kernel version $KERNEL_VERSION...${NC}"
    download_kernel "$KERNEL_VERSION" || {
        echo -e "${RED}Failed to download kernel! Aborting initialization.${NC}"
        return 1
    }
    
    # Step 2: Build kernel
    echo -e "${YELLOW}[2/4] Building kernel for $ARCH...${NC}"
    build_kernel || {
        echo -e "${RED}Failed to build kernel! Aborting initialization.${NC}"
        return 1
    }
    
    # Step 3: Create rootfs
    echo -e "${YELLOW}[3/4] Creating rootfs with $ROOTFS_METHOD...${NC}"
    make_rootfs "$ROOTFS_METHOD" "$ROOTFS_TYPE" || {
        echo -e "${RED}Failed to create rootfs! Aborting initialization.${NC}"
        return 1
    }
    
    # Step 4: Summarize and provide execution command
    echo -e "${YELLOW}[4/4] Environment initialization complete!${NC}"
    echo -e "${YELLOW}-----------------------------------------------${NC}"
    echo -e "${GREEN}Kernel development environment initialized for $ARCH:${NC}"
    echo -e "  Kernel source:      $BASE_DIR/src/kernel"
    echo -e "  Kernel image:       $BASE_DIR/build/kernel-$ARCH"
    echo -e "  Rootfs directory:   $BASE_DIR/rootfs-$ROOTFS_METHOD"
    
    # Determine rootfs method directory
    ROOTFS_DIR=""
    case "$ROOTFS_METHOD" in
        debootstrap) ROOTFS_DIR="$BASE_DIR/rootfs-debootstrap" ;;
        busybox)    ROOTFS_DIR="$BASE_DIR/rootfs-busybox"    ;;
        download)   ROOTFS_DIR="$BASE_DIR/rootfs-download"   ;;
    esac
    
    # Create rootfs image (if not exists)
    if [ ! -f "$BASE_DIR/build/rootfs-$ARCH.cpio.gz" ]; then
        echo -e "${YELLOW}Creating rootfs image...${NC}"
        cd "$ROOTFS_DIR" || return 1
        
        # Temporarily copy kernel modules if they exist
        if [ -d "$BASE_DIR/build/modules" ] && [ -n "$(ls -A "$BASE_DIR/build/modules")" ]; then
            echo -e "${YELLOW}Copying kernel modules to rootfs...${NC}"
            mkdir -p lib/modules
            cp "$BASE_DIR/build/modules"/* lib/modules/
        fi
        
        # Create cpio archive
        find . | cpio -o -H newc > "$BASE_DIR/build/rootfs-$ARCH.cpio.gz"
        
        # Clean up temporary module copies
        if [ -d "lib/modules" ]; then
            rm -rf lib/modules/*
        fi
    fi
    
    # Provide command to run the kernel
    echo -e "${YELLOW}Use the following command to run the kernel:${NC}"
    echo -e "  ${GREEN}./kernel_dev.sh --run-kernel${NC}"
    
    return 0
}

# Display help
show_help() {
    echo -e "${YELLOW}Kernel Development Environment Script Help:${NC}"
    echo "Usage: ./kernel_dev.sh [option] [arguments]"
    echo "Options:"
    echo "  --init ARCH              Download kernel and rootfs and compile, Target architecture (x86, x86_64, arm32, aarch64)"
    echo "  --set-arch ARCH          Set target architecture (x86, x86_64, arm32, aarch64)"
    echo "  --download-kernel [VER]  Download kernel source (default: 5.15.0)"
    echo "  --build-kernel           Build the kernel for current arch"
    echo "  --rebuild-kernel         Rebuild the kernel"
    echo "  --make-rootfs METHOD     Create rootfs (debootstrap, busybox, download)"
    echo "  --build-module DIR       Build kernel module in DIR"
    echo "  --install-module MOD     Install module to rootfs"
    echo "  --run-kernel [MEM] [DIR]  Run kernel in QEMU (default memory: 512M, optional shared directory)"
    echo "  --help                   Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./kernel_dev.sh --set-arch arm32"
    echo "  ./kernel_dev.sh --download-kernel 6.1.0"
    echo "  ./kernel_dev.sh --make-rootfs busybox"
    echo "  ./kernel_dev.sh --make-rootfs debootstrap bullseye"
    echo "  ./kernel_dev.sh --run-kernel
  ./kernel_dev.sh --run-kernel 1G
  ./kernel_dev.sh --run-kernel 512M /path/to/share"
    echo "  ./kernel_dev.sh --init x86_64"
    echo "  ./kernel_dev.sh --init arm32 6.1.0 debootstrap bullseye"
    echo "  ./kernel_dev.sh --init aarch64 5.15.0 download full"
}

# Main script execution
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

case "$1" in
     --init)
        if [ $# -lt 2 ]; then
            echo -e "${RED}Please specify architecture for initialization!${NC}"
            show_help
            exit 1
        fi
        init_environment "$2" "${3:-5.15.0}" "${4:-busybox}" "${5:-minimal}"
        ;;
    --set-arch)
        set_architecture "$2"
        ;;
    --download-kernel)
        if [ $# -ge 2 ]; then
            download_kernel "$2"
        else
            download_kernel
        fi
        ;;
    --build-kernel)
        build_kernel
        ;;
    --rebuild-kernel)
        rebuild_kernel
        ;;
    --make-rootfs)
        if [ $# -lt 2 ]; then
            echo -e "${RED}Please specify rootfs creation method!${NC}"
            show_help
            exit 1
        fi
        make_rootfs "$2" "${3:-}"
        ;;
    --build-module)
        if [ $# -lt 2 ]; then
            echo -e "${RED}Please specify module directory!${NC}"
            show_help
            exit 1
        fi
        build_module "$2"
        ;;
    --install-module)
        if [ $# -lt 2 ]; then
            echo -e "${RED}Please specify module name!${NC}"
            show_help
            exit 1
        fi
        install_module "$2"
        ;;
    --run-kernel)
        case $# in
            1)
                run_kernel
                ;;
            2)
                run_kernel "$2"
                ;;
            3)
                run_kernel "$2" "$3"
                ;;
            *)
                echo -e "${RED}Too many arguments for --run-kernel!${NC}"
                show_help
                exit 1
                ;;
        esac
        ;;
    --help)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
esac

exit 0
