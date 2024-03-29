# EXT_device_openwf

Name

    EXT_device_openwf
    EXT_output_openwf

Name Strings

    EGL_EXT_device_openwf
    EGL_EXT_output_openwf

Contributors

    Daniel Kartch
    James Jones
    Christopher James Halse Rogers

Contacts

    Daniel Kartch, NVIDIA (dkartch 'at' nvidia.com)

Status

    Complete

Version

    Version 5 - January 21, 2016

Number

    EGL Extension #80

Extension Type

    EGL device extension for EGL_EXT_device_openwf

    EGL display extension for EGL_EXT_output_openwf

Dependencies

    EGL_EXT_device_openwf requires EGL_EXT_device_base.

    EGL_EXT_output_openwf requires EGL_EXT_output_base.

    Both require OpenWF Display

    EGL_EXT_device_openwf interacts with EGL_EXT_platform_device

    An EGLDisplay supporting EGL_EXT_output_openwf must be associated
    with an EGLDevice supporting EGL_EXT_device_openwf.

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
    EGL and OpenWF Display. An EGL implementation which provides these
    extensions must have access to sufficient knowledge of the OpenWF
    implementation to be able to perform these mappings. No requirements
    are imposed on how this information is obtained, nor does this
    support have any implications for how EGL devices and outputs are
    implemented. An implementation which supports these extensions may
    support other low level device interfaces, such as DRM/KMS, as well.

New Types

    None

New Procedures and Functions

    None

New Tokens

    Added by EXT_device_openwf:

        Accepted as the <attribute> parameter of eglQueryDeviceAttribEXT

        EGL_OPENWF_DEVICE_ID_EXT                0x3237

        If EGL_EXT_platform_device is present, the following is accepted
        in the <attrib_list> of eglGetPlatformDisplayEXT().

        EGL_OPENWF_DEVICE_EXT                   0x333D

    Added by EXT_output_openwf:

        Accepted in the <attrib_list> of eglGetOutputLayersEXT and as
        the <attribute> parameter of eglQueryOutputLayerAttribEXT

        EGL_OPENWF_PIPELINE_ID_EXT              0x3238

        Accepted in the <attrib_list> of eglGetOutputPortsEXT and as
        the <attribute> parameter of eglQueryOutputPortAttribEXT

        EGL_OPENWF_PORT_ID_EXT                  0x3239

New Behavior for EXT_device_openwf

    EGLDeviceEXTs may be mapped to OpenWF Display devices.

    To obtain a WFD_DEVICE_ID for an EGLDeviceEXT, call
    eglQueryDeviceAtribEXT with <attribute> set to
    EGL_OPENWF_DEVICE_ID_EXT.

If EGL_EXT_platform_device is present, replace the last sentence of the
third paragraph in section 3.2 "Initialization" with the following:

    When <platform> is EGL_PLATFORM_DEVICE_EXT, the only valid attribute
    name is EGL_OPENWF_DEVICE_EXT.  If specified, the value must be a
    WFDDevice created with the device ID returned by querying
    EGL_OPENWF_DEVICE_ID_EXT from the specified EGLDevice.  If the device
    handle does not refer to the correct OpenWF device the behavior is
    undefined.  Calls to eglGetPlatformDeviceEXT() with the same values
    for <platform> and <native_display> but distinct EGL_OPENWF_DEVICE_EXT
    values will return separate EGLDisplays.

    EGL may require the use of the OpenWF device beyond the duration of
    the call to eglGetPlatformDisplayEXT().  The application must ensure
    the device handle remains valid for the lifetime of the display
    returned.  If no OpenWF device handle is specified and EGL requires
    one, it will attempt to create the device itself.  Applications
    should only need to specify an OpenWF device in situations where EGL
    may fail to create one itself due to an existing instance of the same
    underlying device in the process.

New Behavior for EXT_output_openwf

    OpenWF pipeline and port IDs may be used to restrict EGL output
    handle searches and may be queried from EGL output handles.

    Add to Table 3.10.3.1 in EGL_EXT_output_base:

        Attribute                   Type      Access
        --------------------------  -------   ------
        EGL_OPENWF_PIPELINE_ID_EXT  integer   S|R

    Add to Table 3.10.3.2 in EGL_EXT_output_base:

        Attribute                   Type      Access
        --------------------------  -------   ------
        EGL_OPENWF_PORT_ID_EXT      integer   S|R

Issues

    1.  Although the overview says that we do not impose any
        restrictions on how the features are implemented, restrictions
        in the OpenWF specification combined with the chosen interface
        here do implicitly impose limitations. Specifically, the
        wfdCreate* functions can only be called once to obtain OpenWF
        handles. This means that an EGLDevice/Output implementation
        cannot be layered on top of OpenWF without preventing the
        application from calling these functions. So we must assume that
        the implementation instead has some backdoor into OpenWF to
        obtain the object IDs. Possible resolutions include:
        a)  Keep the access model as is. This assumption is a reasonable
            one.
        b)  Flip the requirement. The EGL device/output implementation
            should always create the OpenWF handles itself. We can add
            queries so that the application can get these handles from
            EGL.
        c)  Generalize this extension to support both models. The
            application would have to first query EGL to determine
            whether or not it owns the handles, and then be prepared to
            either query them from EGL or create them itself.
        d)  Require the application to provide its OpenWF device handle
            if it has one.

        RESOLVED: (d), though implementations are free to use (a) when
        possible.

    2.  Should different values of EGL_OPENWF_DEVICE_EXT result in separate
        EGLDisplays?

        RESOLVED: Yes.  Consider an application made up of two independent
        modules running in two independently scheduled threads.  Each
        module calls eglGetPlatformDisplayEXT():

          WFDDevice wfdDev = wfdCreateDevice(WFD_DEFAULT_DEVICE_ID, NULL);
          int attr1[] = { EGL_OPENWF_DEVICE_EXT, wfdDev };
          dpy1 = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                          eglDev,
                                          attr1);

        ...
                                
          dpy2 = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT,
                                          eglDev,
                                          NULL);

        Presumably, if dpy1 == dpy2, they would both be using the same
        WFDDevice for output operations internally.  That would mean
        output operations would likely fail if dpy2 happened to be created
        before dpy1.  This would be painful to debug.  If dpy2 != dpy1,
        failure for dpy2 would be consistent and obvious.  The application
        author would be required to work out a scheme to share the WFDDevice
        between modules before creating EGL displays.

Revision History:

    #5  (January 21st, 2016) James Jones
        - Add EGL_OPENWF_DEVICE_EXT to resolve issue 1.
        - Added possible solution (d) to issue 1, and resolve to use it.
        - Added issue 2.

    #4  (August 22nd, 2014) James Jones
        - Marked complete.
        - Listed Daniel as the contact.

    #3  (June 5th, 2014) Daniel Kartch
        - Assign enumerated values for constants.

    #2  (May 28th, 2014) Daniel Kartch
        - Simplified description of new behavior based on refinements
          to EGL_EXT_output_base.

    #1  (January 31st, 2014) Daniel Kartch
        - Initial draft, representing a signficant reworking of
          functionality previously proposed in
          EGL_EXT_native_device_openwf.
