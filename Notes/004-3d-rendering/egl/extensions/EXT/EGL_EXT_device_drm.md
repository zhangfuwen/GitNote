# EXT_device_drm

Name

    EXT_device_drm
    EXT_output_drm

Name Strings

    EGL_EXT_device_drm
    EGL_EXT_output_drm

Contributors

    Daniel Kartch
    James Jones
    Christopher James Halse Rogers

Contacts

    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Status

    Complete

Version

    Version 5 - December 28th, 2015

Number

    EGL Extension #79

Extension Type

    EGL device extension for EGL_EXT_device_drm

    EGL display extension for EGL_EXT_output_drm

Dependencies

    EGL_EXT_device_drm requires EGL_EXT_device_base.

    EGL_EXT_device_drm interacts with EGL_EXT_platform_device

    EGL_EXT_device_drm requires a DRM driver.

    EGL_EXT_output_drm requires EGL_EXT_output_base.

    EGL_EXT_output_drm requires a DRM driver which supports KMS.

    An EGLDisplay supporting EGL_EXT_output_drm must be associated
    with an EGLDevice supporting EGL_EXT_device_drm.

Overview

    Increasingly, EGL and its client APIs are being used in place of
    "native" rendering APIs to implement the basic graphics
    functionality of native windowing systems.  This creates demand
    for a method to initialize EGL displays and surfaces directly on
    top of native GPU or device objects rather than native window
    system objects.  The mechanics of enumerating the underlying
    native devices and constructing EGL displays and surfaces from
    them have been solved in various platform and implementation-
    specific ways.  The EGL device family of extensions offers a
    standardized framework for bootstrapping EGL without the use of
    any underlying "native" APIs or functionality.

    These extensions define how to map device and output handles between
    EGL and DRM/KMS. An EGL implementation which provides these
    extensions must have access to sufficient knowledge of the DRM
    implementation to be able to perform these mappings. No requirements
    are imposed on how this information is obtained, nor does this
    support have any implications for how EGL devices and outputs are
    implemented. Among the possibilities, support may be implemented in
    a generic fashion by layering on top of DRM, or EGL and DRM backends
    may be provided by the same vendor and share privileged lower level
    resources. An implementation which supports these extensions may
    support other low level device interfaces, such as OpenWF Display,
    as well.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Added by EXT_device_drm:

        Accepted as the <name> parameter of eglQueryDeviceStringEXT

        EGL_DRM_DEVICE_FILE_EXT                 0x3233

        If EGL_EXT_platform_device is present, the following is accepted
        in the <attrib_list> of eglGetPlatformDisplayEXT().

        EGL_DRM_MASTER_FD_EXT                   0x333C

    Added by EXT_output_drm:

        Accepted in the <attrib_list> of eglGetOutputLayersEXT and as
        the <attribute> parameter of eglQueryOutputLayerAttribEXT

        EGL_DRM_CRTC_EXT                        0x3234
        EGL_DRM_PLANE_EXT                       0x3235

        Accepted in the <attrib_list> of eglGetOutputPortsEXT and as
        the <attribute> parameter of eglQueryOutputPortAttribEXT

        EGL_DRM_CONNECTOR_EXT                   0x3236

New Behavior for EXT_device_drm

    EGLDeviceEXTs may be mapped to DRM device files.

    To obtain a DRM device file for an EGLDeviceEXT, call
    eglQueryDeviceStringEXT with <name> set to EGL_DRM_DEVICE_FILE_EXT.
    The function will return a pointer to a string containing the name
    of the device file (e.g. "/dev/dri/cardN").

If EGL_EXT_platform_device is present, replace the last sentence of the
third paragraph in section 3.2 "Initialization" with the following:

    When <platform> is EGL_PLATFORM_DEVICE_EXT, the only valid attribute
    name is EGL_DRM_MASTER_FD_EXT.  If specified, the value must be a file
    descriptor with DRM master permissions on the DRM device associated
    with the specified EGLDevice, as determined by EGL_DRM_DEVICE_FILE_EXT.
    If the file descriptor does not refer to the correct DRM device or
    does not have master permissions, the behavior is undefined.  Calls to
    eglGetPlatformDeviceEXT() with the same values for <platform> and
    <native_display> but distinct EGL_DRM_MASTER_FD_EXT values will return
    separate EGLDisplays.

    If EGL requires the use of the DRM file descriptor beyond the duration
    of the call to eglGetPlatformDispay(), it will duplicate it.  If no
    file descriptor is specified and EGL requires one, it will attempt to
    open the device itself.  Applications should only need to specify a
    file descriptor in situations where EGL may fail to open a file
    descriptor itself, generally due to lack of permissions, or when EGL
    will fail to acquire DRM master permissions due to conflicts with an
    existing DRM client.  DRM master permissions are only required when EGL
    must modify output attributes.  This extension does not define any
    situations in which output attributes will be modified.

New Behavior for EXT_output_drm

    KMS CRTC, plane, and connector IDs may be used to restrict EGL
    output handle searches and may be queried from EGL output handles.

    Add to Table 3.10.3.1 in EGL_EXT_output_base:

        Attribute               Type      Access
        ---------------------   -------   ------
        EGL_DRM_CRTC_EXT        integer   S|R
        EGL_DRM_PLANE_EXT       integer   S|R

    Add to Table 3.10.3.2 in EGL_EXT_output_base:

        Attribute               Type      Access
        ---------------------   -------   ------
        EGL_DRM_CONNECTOR_EXT   integer   S|R

    Add to description of eglOutputLayerAttribEXT:

        If <layer> corresponds to a KMS CRTC and <attribute> is
        EGL_DRM_PLANE_EXT, or if <layer> corresponds to a KMS plane and
        <attribute> is EGL_DRM_CRTC_EXT, an EGL_BAD_MATCH error is
        generated.

Issues

    1)  Should different values of EGL_DRM_MASTER_FD_EXT result in separate
        EGLDisplays?

        RESOLVED: Yes.  Consider an application made up of two independent
        modules running in two independently scheduled threads.  Each
        module calls eglGetPlatformDisplayEXT():

          int fd = open("/dev/dri/card0", O_RDWR);
          int attr1[] = { EGL_DRM_MASTER_FD_EXT, fd };
          dpy1 = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                          eglDev,
                                          attr1);

        ...
                                
          dpy2 = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                          eglDev,
                                          NULL);

        Presumably, if dpy1 == dpy2, they would both be using the same DRM
        fd for output operations internally.  That would mean display
        attribute updates would likely fail if dpy2 happened to be created
        before dpy1.  This would be painful to debug.  If dpy2 != dpy1,
        failure for dpy2 would be consistent and obvious.  The application
        author would be required to work out a scheme to share the master
        FD between modules before creating EGL displays.
   
Revision History:

    #5  (December 28th, 2015) James Jones
        - Added EGL_DRM_MASTER_FD_EXT and associated
          language.
        - Added issue 1.

    #4  (August 22nd, 2014) James Jones
        - Marked complete.
        - Listed Daniel as the contact.

    #3  (June 5th, 2014) Daniel Kartch
        - Assigned enumerated values for constants.

    #2  (May 28th, 2014) Daniel Kartch
        - Simplified description of new behavior based on refinements
          to EGL_EXT_output_base.

    #1  (January 31st, 2014) Daniel Kartch
        - Initial draft, representing a signficant reworking of
          functionality previously proposed in
          EGL_EXT_native_device_drm.
