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
    
    # Assume we have a config file; adjust as needed
    if [ ! -f ".config" ]; then
        echo -e "${YELLOW}No kernel config found, using default for $ARCH...${NC}"
        make "$ARCH_PARAM"_defconfig
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
    cat > "$ROOTFS_DIR/etc/inittab" << EOF
::sysinit:/etc/init.d/rcS
::respawn:/sbin/getty -L tty1 9600
::respawn:/sbin/getty -L tty2 9600
::respawn:/sbin/getty -L tty3 9600
::respawn:/sbin/getty -L tty4 9600
EOF
    
    cat > "$ROOTFS_DIR/etc/init.d/rcS" << EOF
#!/bin/sh
mount -a
mkdir -p /dev/pts
mount -t devpts devpts /dev/pts
echo "Starting $ARCH rootfs..."
exec /sbin/init
EOF
    chmod +x "$ROOTFS_DIR/etc/init.d/rcS"
    
    echo -e "${GREEN}Rootfs created with debootstrap successfully!${NC}"
    echo -e "${GREEN}Rootfs located at: $ROOTFS_DIR${NC}"
    return 0
}

# Function: Create rootfs with busybox
make_rootfs_busybox() {
    local BUSYBOX_VERSION="${2:-1.35.0}"
    echo -e "${YELLOW}Creating rootfs with busybox for $ARCH...${NC}"
    
    # Create rootfs directory if it doesn't exist
    mkdir -p "$BASE_DIR/rootfs-busybox"
    ROOTFS_DIR="$BASE_DIR/rootfs-busybox"
    
    # Check if busybox is installed or build it
    if ! command -v busybox &> /dev/null; then
        echo -e "${YELLOW}Busybox not found, building from source...${NC}"
        build_busybox "$BUSYBOX_VERSION"
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to build busybox!${NC}"
            return 1
        fi
    fi
    
    # Clean previous rootfs
    rm -rf "$ROOTFS_DIR"/*
    
    # Create basic rootfs structure
    mkdir -p "$ROOTFS_DIR"{bin,sbin,etc,lib,lib64,dev,proc,sys,home,root,tmp,var}
    
    # Install busybox
    BUSYBOX_BINARY="$BASE_DIR/src/busybox/busybox"
    if [ -f "$BUSYBOX_BINARY" ]; then
        cp "$BUSYBOX_BINARY" "$ROOTFS_DIR/bin/"
    else
        cp "$(which busybox)" "$ROOTFS_DIR/bin/"
    fi
    
    # Create symlinks for busybox commands
    cd "$ROOTFS_DIR/bin" || return 1
    ln -s busybox sh
    for cmd in $(busybox --list); do
        ln -s busybox "$cmd"
    done
    
    # Create necessary files
    cat > "$ROOTFS_DIR/etc/fstab" << EOF
proc            /proc           proc    defaults        0 0
sysfs           /sys            sysfs   defaults        0 0
devtmpfs        /dev            devtmpfs mode=0755,size=1024M 0 0
EOF
    
    cat > "$ROOTFS_DIR/etc/inittab" << EOF
::sysinit:/etc/init.d/rcS
::respawn:/sbin/getty -L tty1 9600
::respawn:/sbin/getty -L tty2 9600
::respawn:/sbin/getty -L tty3 9600
::respawn:/sbin/getty -L tty4 9600
EOF
    
    mkdir -p "$ROOTFS_DIR/etc/init.d"
    cat > "$ROOTFS_DIR/etc/init.d/rcS" << EOF
#!/bin/sh
mount -a
mkdir -p /dev/pts
mount -t devpts devpts /dev/pts
echo "Starting $ARCH rootfs..."
exec /sbin/init
EOF
    chmod +x "$ROOTFS_DIR/etc/init.d/rcS"
    
    echo -e "${GREEN}Rootfs created with busybox successfully!${NC}"
    echo -e "${GREEN}Rootfs located at: $ROOTFS_DIR${NC}"
    return 0
}

# Function: Build busybox from source
build_busybox() {
    local VERSION="$1"
    local BUSYBOX_ARCHIVE="busybox-$VERSION.tar.bz2"
    local BUSYBOX_URL="https://busybox.net/downloads/$BUSYBOX_ARCHIVE"
    local BUSYBOX_SRC_DIR="$BASE_DIR/src/busybox"
    
    echo -e "${YELLOW}Building busybox version $VERSION...${NC}"
    
    # Create source directory if it doesn't exist
    mkdir -p "$BASE_DIR/src"
    
    cd "$BASE_DIR/src" || { echo -e "${RED}Failed to access source directory!${NC}"; return 1; }
    
    # Download busybox archive
    if [ ! -f "$BUSYBOX_ARCHIVE" ]; then
        if command -v wget &> /dev/null; then
            wget -c "$BUSYBOX_URL"
        else
            curl -L -O "$BUSYBOX_URL"
        fi
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download busybox!${NC}"
            return 1
        fi
    fi
    
    # Extract busybox source
    if [ -d "$BUSYBOX_SRC_DIR" ]; then
        rm -rf "$BUSYBOX_SRC_DIR"
    fi
    
    tar -xjf "$BUSYBOX_ARCHIVE"
    mv "busybox-$VERSION" "$BUSYBOX_SRC_DIR"
    
    # Set architecture and build busybox
    cd "$BUSYBOX_SRC_DIR" || return 1
    
    case "$ARCH" in
        x86)
            make defconfig
            ;;
        x86_64)
            make defconfig
            ;;
        arm32)
            make arm_defconfig
            ;;
        aarch64)
            make arm64_defconfig
            ;;
    esac
    
    make -j$(nproc)
    make install
    
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
    mkdir -p "$BASE_DIR/modules"
    
    # Check if module directory exists
    if [ ! -d "$BASE_DIR/modules/$MODULE_DIR" ]; then
        echo -e "${RED}Module directory not found!${NC}"
        return 1
    fi
    
    cd "$BASE_DIR/modules/$MODULE_DIR" || return 1
    
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
    MODULE_NAME=$(basename "$(find . -name '*.ko' 2>/dev/null | head -n 1)")
    if [ -n "$MODULE_NAME" ]; then
        cp "$MODULE_NAME" "$BASE_DIR/build/modules/"
        echo -e "${GREEN}Kernel module $MODULE_NAME built successfully!${NC}"
    else
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
    # Verify kernel exists
    KERNEL_IMAGE="$BASE_DIR/build/kernel-$ARCH"
    if [ ! -f "$KERNEL_IMAGE" ]; then
        echo -e "${RED}Kernel image for $ARCH not found! Please build the kernel first.${NC}"
        return 1
    fi
    
    # Verify rootfs exists
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
    
    echo -e "${YELLOW}Running kernel for $ARCH...${NC}"
    
    # Create rootfs image (example using cpio)
    cd "$ROOTFS_DIR" || return 1
    find . | cpio -o -H newc > "$BASE_DIR/build/rootfs-$ARCH.cpio.gz"
    
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
    qemu-system-"$ARCH" \
        -machine "$MACHINE_TYPE" \
        -kernel "$KERNEL_IMAGE" \
        -initrd "$BASE_DIR/build/rootfs-$ARCH.cpio.gz" \
        -m 512M \
        -nographic \
        -append "root=/initrd console=ttyS0"
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
        find . | cpio -o -H newc > "$BASE_DIR/build/rootfs-$ARCH.cpio.gz"
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
    echo "  --run-kernel             Run kernel in QEMU"
    echo "  --help                   Display this help message"
    echo ""
    echo "Examples:"
    echo "  ./kernel_dev.sh --set-arch arm32"
    echo "  ./kernel_dev.sh --download-kernel 6.1.0"
    echo "  ./kernel_dev.sh --make-rootfs busybox"
    echo "  ./kernel_dev.sh --make-rootfs debootstrap bullseye"
    echo "  ./kernel_dev.sh --run-kernel"
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
        run_kernel
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
