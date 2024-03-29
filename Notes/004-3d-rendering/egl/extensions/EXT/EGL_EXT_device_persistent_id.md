# EXT_device_persistent_id

Name

    EXT_device_persistent_id

Name Strings

    EGL_EXT_device_persistent_id

Contributors

    Kyle Brenneman,  NVIDIA  (kbrenneman 'at' nvidia.com)

Contact

    Kyle Brenneman,  NVIDIA  (kbrenneman 'at' nvidia.com)

Version

    Version 1 - April 19, 2021

Number

    EGL Extension #142

Extension Type

    EGL device extension

Dependencies

    Written based on the wording of the EGL 1.5 specification.

    EGL_EXT_device_query is required.

Overview

    Applications can query a list of EGLDeviceEXT handles, but those
    handles are only valid within the process that queried them. An
    application has no way, for example, to record its selection and
    select the same device when run again later.

    This extension provides a vendor name and a set of UUID's, which
    provide a unique, persistent identifier for EGLDeviceEXT handles.
    This allows applications to find the EGLDeviceEXT for the same
    device across multiple processes, and across multiple APIs.

New Procedures and Functions

    EGLBoolean eglQueryDeviceBinaryEXT(EGLDeviceEXT device,
                                      EGLint name,
                                      EGLint max_size,
                                      void *value,
                                      EGLint *size);

New Tokens

    Accepted by the <name> parameter of eglQueryDeviceStringEXT:

        EGL_DRIVER_NAME_EXT          0x335E

    Accepted by the <name> parameter of eglQueryDeviceBinaryEXT:

        EGL_DEVICE_UUID_EXT          0x335C
        EGL_DRIVER_UUID_EXT          0x335D

Changes to section 3.2 (Devices)

    Add the following paragraph to the description of
    eglQueryDeviceStringEXT:

    EGL_DRIVER_NAME_EXT returns a string which identifies the driver
    that controls the device. This string remains persistent across
    multiple versions of a driver, and an application can use strcmp(3)
    to compare the strings for equality. Otherwise, the contents are
    implementation-defined.


    Add to the end of section 3.2:

    To query a binary attribute for a device, use:

        EGLBoolean eglQueryDeviceBinaryEXT(EGLDeviceEXT device,
                                          EGLint name,
                                          EGLint max_size,
                                          void *value,
                                          EGLint *size);

    On success, EGL_TRUE is returned. If <value> is NULL, then
    <max_size> is ignored, and the size of the attribute in bytes is
    returned in <size>.

    On failure, EGL_FALSE is returned. An EGL_BAD_ATTRIBUTE error is
    generated if <name> is not a valid attribute. An EGL_BAD_DEVICE_EXT
    error is generated if <device> is not a valid EGLDeviceEXT.

    If <value> is not NULL, then the attribute value is returned in
    <value>. At most <max_size> bytes are written. <size> returns the
    number of bytes that were actually written.

    Note that the EGL_DEVICE_UUID_EXT and EGL_DRIVER_UUID_EXT attributes
    are always 16-byte values, and so the application can simply use a
    16-byte buffer without needing to query the size beforehand. Future
    extensions may add variable-length attributes.


    EGL_DEVICE_UUID_EXT is a UUID that identifies a physical device,
    returned as a 16-byte binary value. The device UUID uniquely
    identifies a physical device, and is persistent across reboots,
    processes, APIs, and (to the extent possible) driver versions.
    
    EGL_DEVICE_UUID_EXT may or may not be persistent across changes in
    hardware configuration. Similarly, it is not guaranteed to be unique
    or persistent across different (physical or virtual) computers.

    Note that EGL_DEVICE_UUID_EXT alone is not guaranteed to be unique
    across all EGLDeviceEXT handles. If an EGL implementation supports
    multiple drivers, and two drivers can use the same physical device,
    then there will be a separate EGLDeviceEXT handle from each driver.
    Both EGLDeviceEXT handles may use the same device UUID.

    In that case, an application must use EGL_DRIVER_NAME_EXT or
    EGL_DRIVER_UUID_EXT to distinguish between the two EGLDeviceEXT
    handles.


    EGL_DRIVER_UUID_EXT is a UUID that identifies a driver build
    in use for a device. The driver UUID is persistent across reboots,
    processes, and APIs, but is not persistent across driver versions.

Issues

    1.  Should we use UUID's or strings to identify devices?

        RESOLVED: Use UUID's for devices, plus a vendor name string to
        disambiguate devices that are supported by multiple drivers.

        A device UUID and driver UUID allow an application to correlate
        an EGLDeviceEXT with the same device in other APIs, such as a
        VkPhysicalDevice in Vulkan.

        A UUID does not impose any additional requirements on an EGL
        implementation compared to a string: If an EGL implementation
        could generate a string identifier, then the implementation can
        simply hash that string to generate a UUID value.

    2.  Can two EGLDeviceEXT handles have the same EGL_DEVICE_UUID_EXT?

        RESOLVED: Yes, if they correspond to the same physical device.

        The semantics of the device and driver UUID's are inherited from
        Vulkan, which only requires that a device UUID be unique to a
        physical device, not unique across VkPhysicalDevice handles.

    3.  Do we need the EGL_DRIVER_NAME_EXT string?

        RESOLVED: Yes, because the EGL_DEVICE_UUID_EXT alone is not
        unique, and EGL_DRIVER_UUID_EXT is not persistent.

        A (EGL_DRIVER_NAME_EXT, EGL_DEVICE_UUID_EXT) pair provides a
        unique, persistent identifier.

        In addition, on systems that use libglvnd, applications could
        use EGL_DRIVER_NAME_EXT to match the vendor names from
        GLX_EXT_libglvnd.

    4.  What happens if an application stores a device UUID, and the
        hardware configuration or driver version changes?

        RESOLVED: The device UUID may become invalid, and the
        application should select a new device.

        If a device is removed from a system, then there will be no
        EGLDeviceEXT handle for it, and thus no device UUID for it.

        Similarly, if a device is moved within a system (e.g., plugged
        into a different PCI slot), then a driver may not be able to
        identify it as the same device, and so the device might get a
        different UUID.

        While not a requirement, drivers should still try to keep device
        UUID's persistent whenever possible, to avoid invalidating
        config files. Similarly, if a device is removed or replaced,
        then a driver should try to ensure that the same device UUID
        does not refer to a different device.

        As an example, a driver could derive a UUID based on a PCI
        vendor and device number, plus the PCI domain, bus, slot, and
        function numbers:

        * The PCI device number ensures that replacing a GPU with a
          different model in the same PCI slot produces a different
          device UUID string.
        * Using the PCI bus numbers ensures that two identical
          GPU's in the same system have unique UUID's.
        * The whole tuple can easily stay persistent across driver
          versions.

Revision History

    #1 (April 19, 2021) Kyle Brenneman

        - Initial draft
