#!/bin/bash

# Docker to QEMU Image Exporter
# Exports a Docker image to a QEMU-compatible disk image and runs it with QEMU
# Usage: ./docker2qemu.sh [OPTIONS] IMAGE_NAME

# Colors for terminal output
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

# Defaults
IMAGE_SIZE=""
QEMU_IMAGE="docker-export.qcow2"
CONTAINER_TOOL="docker"
MOUNT_DIR="/tmp/docker2qemu-mount"
WORK_DIR="/tmp/docker2qemu-work"
RUN_QEMU=false
KERNEL_IMAGE=""
ARCH="x86_64"

# Function: Check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        return 1
    fi
    return 0
}

# Function: Install required packages
install_packages() {
    echo -e "${YELLOW}Installing required packages...${NC}"
    
    if [ "$(lsb_release -is)" == "Ubuntu" ]; then
        sudo apt-get update || return 1
        sudo apt-get install -y qemu-utils fuse-overlayfs podman || return 1
    elif [ "$(rpm --query centos-release &> /dev/null || echo 0)" -ne 0 ]; then
        sudo dnf install -y qemu-utils fuse-overlayfs podman || return 1
    else
        echo -e "${RED}Unsupported Linux distribution. Please install qemu-utils, fuse-overlayfs, and podman manually.${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Packages installed successfully!${NC}"
    return 0
}

# Function: Check and install container tool (Docker or Podman)
check_container_tool() {
    if check_command "docker"; then
        echo -e "${YELLOW}Docker found, using Docker.${NC}"
        CONTAINER_TOOL="docker"
    elif check_command "podman"; then
        echo -e "${YELLOW}Podman found, using Podman.${NC}"
        CONTAINER_TOOL="podman"
    else
        echo -e "${YELLOW}Neither Docker nor Podman found, installing Podman...${NC}"
        
        if [ "$(lsb_release -is)" == "Ubuntu" ]; then
            sudo apt-get install -y podman || return 1
        elif [ "$(rpm --query centos-release &> /dev/null || echo 0)" -ne 0 ]; then
            sudo dnf install -y podman || return 1
        else
            echo -e "${RED}Failed to install Podman on your distribution. Please install Docker or Podman manually.${NC}"
            return 1
        fi
        
        CONTAINER_TOOL="podman"
        echo -e "${GREEN}Podman installed successfully!${NC}"
    fi
    
    return 0
}

# Function: Estimate image size based on Docker image
estimate_image_size() {
    local image=$1
    echo -e "${YELLOW}Estimating required disk space for image: $image${NC}"
    
    local image_size=$(eval "$CONTAINER_TOOL images --format '{{.Size}}' $image")
    if [ -z "$image_size" ]; then
        echo -e "${RED}Failed to get image size. Using default 10GB.${NC}"
        return 10GB
    fi
    
    # Convert size to GB (e.g., 500MB -> 1GB, 1.5GB -> 2GB)
    local size_num=$(echo "$image_size" | grep -o '[0-9\.]*' | head -n 1)
    local size_unit=$(echo "$image_size" | grep -o '[A-Za-z]*' | head -n 1)
    
    case "$size_unit" in
        "GB" | "gb")
            # Add 2GB buffer
            local estimated_size=$(echo "($size_num + 2)" | bc)
            echo -e "${GREEN}Estimated size: $estimated_size GB${NC}"
            return $estimated_size"GB"
            ;;
        "MB" | "mb")
            # Convert to GB and add buffer
            local gb_size=$(echo "($size_num / 1024 + 2)" | bc -l | cut -d. -f1)
            echo -e "${GREEN}Estimated size: $gb_size GB${NC}"
            return $gb_size"GB"
            ;;
        *)
            echo -e "${RED}Unsupported size unit: $size_unit. Using default 10GB.${NC}"
            return 10"GB"
            ;;
    esac
}

# Function: Create QEMU image
create_qemu_image() {
    local image=$1
    local size=$2
    
    echo -e "${YELLOW}Creating QEMU image with size: $size${NC}"
    
    # Create the image using qemu-img
    qemu-img create -f qcow2 "$QEMU_IMAGE" "$size"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create QEMU image!${NC}"
        return 1
    fi
    
    echo -e "${GREEN}QEMU image created: $QEMU_IMAGE${NC}"
    return 0
}

# Function: Mount QEMU image
mount_image() {
    local image=$1
    
    echo -e "${YELLOW}Mounting QEMU image...${NC}"
    
    # Create mount directories
    mkdir -p "$MOUNT_DIR" "$WORK_DIR"
    
    # Format and mount the image
    losetup -f || return 1
    local loop_device=$(losetup -f)
    
    losetup "$loop_device" "$image"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to set up loop device!${NC}"
        return 1
    fi
    
    # Check if the image is already formatted
    file -s "$loop_device" | grep -q "ext4"
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Formatting image with ext4...${NC}"
        mkfs.ext4 "$loop_device" || return 1
    fi
    
    mount "$loop_device" "$MOUNT_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to mount image!${NC}"
        losetup -d "$loop_device"
        return 1
    fi
    
    echo -e "${GREEN}Image mounted at: $MOUNT_DIR${NC}"
    return 0
}

# Function: Export Docker image to QEMU image
export_docker_to_qemu() {
    local image=$1
    
    echo -e "${YELLOW}Exporting Docker image: $image to QEMU image${NC}"
    
    # Create a temporary container
    local container_id=$($CONTAINER_TOOL create $image)
    if [ -z "$container_id" ]; then
        echo -e "${RED}Failed to create container from image!${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}Container created: $container_id${NC}"
    
    # Export container filesystem
    $CONTAINER_TOOL export $container_id > "$WORK_DIR/container.tar"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to export container!${NC}"
        $CONTAINER_TOOL rm $container_id
        return 1
    fi
    
    echo -e "${GREEN}Container filesystem exported successfully!${NC}"
    
    # Extract to mounted image
    cd "$MOUNT_DIR" || return 1
    tar -xf "$WORK_DIR/container.tar"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to extract container to image!${NC}"
        $CONTAINER_TOOL rm $container_id
        return 1
    fi
    
    # Create basic system directories
    mkdir -p dev proc sys etc/init.d
    touch etc/inittab
    
    # Clean up
    $CONTAINER_TOOL rm $container_id
    echo -e "${GREEN}Container cleaned up: $container_id${NC}"
    
    echo -e "${GREEN}Docker image successfully exported to QEMU image!${NC}"
    echo -e "${GREEN}QEMU image: $QEMU_IMAGE${NC}"
    echo -e "${GREEN}Mounted directory: $MOUNT_DIR${NC}"
    return 0
}

# Function: Run QEMU with kernel and image
run_qemu() {
    local kernel="$1"
    local image="$2"
    local arch="$3"
    
    echo -e "${YELLOW}Running QEMU with kernel: $kernel and image: $image${NC}"
    
    # Determine QEMU system command based on architecture
    case "$arch" in
        x86|x86_64)
            qemu_cmd="qemu-system-x86_64"
            machine_type="q35"
            kernel_param="kernel"
            ;;
        arm32)
            qemu_cmd="qemu-system-arm"
            machine_type="virt"
            kernel_param="kernel"
            ;;
        aarch64)
            qemu_cmd="qemu-system-aarch64"
            machine_type="virt"
            kernel_param="kernel"
            ;;
        *)
            echo -e "${RED}Unsupported architecture: $arch. Using x86_64.${NC}"
            qemu_cmd="qemu-system-x86_64"
            machine_type="q35"
            kernel_param="kernel"
            ;;
    esac
    
    # Check if QEMU command exists
    if ! command -v "$qemu_cmd" &> /dev/null; then
        echo -e "${RED}QEMU command not found: $qemu_cmd. Please install QEMU.${NC}"
        return 1
    fi
    
    # Run QEMU
    $qemu_cmd \
        -$kernel_param "$kernel" \
        -drive file="$image",format=qcow2,if=virtio \
        -machine "$machine_type" \
        -m 1024 \
        -nographic \
        -append "root=/dev/vda console=ttyS0"
    
    echo -e "${GREEN}QEMU execution completed!${NC}"
    return 0
}

# Function: Get existing QEMU images
get_existing_images() {
    echo "$(find . -maxdepth 1 -name "*.qcow2" -o -name "*.img" | sort)"
}

# Function: Clean up
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    if [ -d "$MOUNT_DIR" ]; then
        umount "$MOUNT_DIR" 2>/dev/null
        rm -rf "$MOUNT_DIR"
    fi
    
    if [ -d "$WORK_DIR" ]; then
        rm -rf "$WORK_DIR"
    fi
    
    # Detach loop device
    losetup -a | grep -q "$MOUNT_DIR"
    if [ $? -eq 0 ]; then
        losetup -d $(losetup -j "$MOUNT_DIR" | cut -d: -f1)
    fi
    
    echo -e "${GREEN}Cleanup completed!${NC}"
}

# Function: Show help
show_help() {
    echo -e "${YELLOW}Docker to QEMU Image Exporter Help:${NC}"
    echo "Usage: ./docker2qemu.sh [OPTIONS] IMAGE_NAME"
    echo "Options:"
    echo "  -s, --size SIZE      Specify QEMU image size (e.g., 10GB, 500MB)"
    echo "  -o, --output FILE    Specify output QEMU image file (default: docker-export.qcow2)"
    echo "  -k, --kernel FILE    Specify kernel image file (for QEMU run)"
    echo "  -a, --arch ARCH      Specify architecture (x86_64, arm32, aarch64, default: x86_64)"
    echo "  -r, --run            Run QEMU after export"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "QEMU Run Examples:"
    echo "  ./docker2qemu.sh -r -k vmlinuz-5.15.0 myimage.qcow2"
    echo "  ./docker2qemu.sh --run --arch arm32 --kernel zImage ubuntu-arm.qcow2"
    echo ""
    echo "Existing QEMU images (auto-completion):"
    get_existing_images | while read -r img; do
        echo "  $img"
    done
}

# Main script execution
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}       Docker to QEMU Image Exporter         ${NC}"
echo -e "${GREEN}==============================================${NC}"

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        -o|--output)
            QEMU_IMAGE="$2"
            shift 2
            ;;
        -k|--kernel)
            KERNEL_IMAGE="$2"
            shift 2
            ;;
        -a|--arch)
            ARCH="$2"
            shift 2
            ;;
        -r|--run)
            RUN_QEMU=true
            shift 1
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            IMAGE_NAME="$1"
            shift 1
            ;;
    esac
done

# Check if image name is provided
if [ -z "$IMAGE_NAME" ]; then
    echo -e "${RED}Please specify a Docker image name!${NC}"
    show_help
    exit 1
fi

# Check if image exists
echo -e "${YELLOW}Checking if image exists: $IMAGE_NAME${NC}"
$CONTAINER_TOOL images | grep -q "$IMAGE_NAME"
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Image not found, pulling from repository...${NC}"
    $CONTAINER_TOOL pull "$IMAGE_NAME"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to pull image: $IMAGE_NAME${NC}"
        exit 1
    fi
    echo -e "${GREEN}Image pulled successfully!${NC}"
fi

# Install required packages
install_packages || exit 1

# Check container tool
check_container_tool || exit 1

# Determine image size
if [ -z "$IMAGE_SIZE" ]; then
    IMAGE_SIZE=$(estimate_image_size "$IMAGE_NAME")
fi

# Create QEMU image
create_qemu_image "$IMAGE_NAME" "$IMAGE_SIZE" || exit 1

# Mount QEMU image
mount_image "$QEMU_IMAGE" || exit 1

# Export Docker image to QEMU
export_docker_to_qemu "$IMAGE_NAME" || exit 1

# Run QEMU if specified
if [ "$RUN_QEMU" = true ]; then
    if [ -z "$KERNEL_IMAGE" ]; then
        echo -e "${RED}Please specify a kernel image with -k or --kernel!${NC}"
        show_help
        exit 1
    fi
    
    if [ ! -f "$KERNEL_IMAGE" ]; then
        echo -e "${RED}Kernel image not found: $KERNEL_IMAGE${NC}"
        exit 1
    fi
    
    if [ ! -f "$QEMU_IMAGE" ]; then
        echo -e "${RED}QEMU image not found: $QEMU_IMAGE${NC}"
        exit 1
    fi
    
    run_qemu "$KERNEL_IMAGE" "$QEMU_IMAGE" "$ARCH"
fi

# Clean up
cleanup

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}Docker to QEMU image export completed!${NC}"
echo -e "${GREEN}QEMU image saved as: $QEMU_IMAGE${NC}"
if [ "$RUN_QEMU" = true ]; then
    echo -e "${GREEN}QEMU execution completed!${NC}"
fi
echo -e "${GREEN}==============================================${NC}"