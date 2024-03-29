# KHR_display_reference

Name

    KHR_display_reference

Name Strings

    EGL_KHR_display_reference

Contributors

    James Jones
    Daniel Kartch

Contacts

    James Jones,  NVIDIA  (jajones 'at' nvidia.com)

Status

    Complete
    Ratified by the Khronos Board of Promoters on March 31, 2017.

Version

    Version 4 - March 15, 2018

Number

    EGL Extension #126

Extension Type

    EGL client extension

Dependencies

    Written based on the wording of the EGL 1.5 specification.

    Requires EGL_EXT_platform_base or EGL 1.5

    Interacts with EGL platform extensions.

    Interacts with the EGL_EXT_device_query extension.

Overview

    The existing semantics of EGLDisplay object lifetimes work well for
    applications in which one module manages all EGL usage, and in which
    EGL displays are expected to remain available until application
    termination once they are instantiated.  However, EGL does not
    provide reasonable semantics in the case where applications rely on
    toolkit libraries which use EGL independently from the application
    itself.

    This issue can be solved by adding a per-EGLDisplay reference
    counter which is incremented by eglInitialize calls. Resource
    destruction can then be deferred until a corresponding number of
    eglTerminate calls is made. However, switching to this behavior
    universally could cause backwards incompatibility problems with
    existing applications that assume a single eglTerminate will
    immediately free resources regardless of how many times the display
    has been initialized.

    We therefore must support both behaviors. A new attribute specified
    when the EGLDisplay is obtained will indicate whether or not
    reference counting is enabled. If an application requests the
    EGLDisplay multiple times with different values for this attribute,
    two separate displays will be returned. The one potential drawaback
    is that these displays will have independent resource spaces, so
    objects allocated from one cannot be used by the other. However, the
    goal here is to support modules that access EGL independently. In
    such a use case, they are not likely to need to share resources with
    another module, particularly one that uses a different method for
    accessing the display.

New Types

    None

New Functions

    EGLBoolean eglQueryDisplayAttribKHR(EGLDisplay dpy,
                                        EGLint name,
                                        EGLAttrib *value);

New Tokens

    Accepted as an attribute in the <attrib_list> parameter of
    eglGetPlatformDisplay and the <name> parameter of
    eglQueryDisplayAttribKHR:

        EGL_TRACK_REFERENCES_KHR                         0x3352

In section "3.2 Initialization":

Remove the sentence in the description of eglGetPlatformDisplay
indicating no valid attribute names are defined, and add the following:

    The EGL_TRACK_REFERENCES_KHR attribute may be set to EGL_TRUE or
    EGL_FALSE to indicate whether or not an EGLDisplay that tracks
    reference counts for eglInitialize and eglTerminate calls (as
    described below) is desired. If not specified, the default is
    platform dependent. Implementations are not required to support both
    EGL_TRUE and EGL_FALSE for this attribute. If separate successful
    calls are made to eglGetPlatformDisplay requesting default and non-
    default behavior for reference counting, two independent EGLDisplays
    will be returned.

Also add to the Errors section:

    An EGL_BAD_ATTRIBUTE error is generated if the requested value for
    EGL_TRACK_REFERENCES_KHR is not supported.

Replace the first sentence of the second paragraph of the description of
eglInitialize with:

    When a previously uninitialized display is initialized, its
    reference count will be set to one. Initializing an already-
    initialized display is allowed, and will return EGL_TRUE and update
    the EGL version numbers, but has no other effect except to increment
    the display's reference count if its EGL_TRACK_REFERENCES_KHR
    attribute is EGL_TRUE.

Insert after the declaration of eglTerminate:

    If the specified display's EGL_TRACK_REFERENCES_KHR attribute is
    EGL_FALSE, eglTerminate will immediately set its reference count
    to zero. Otherwise, its reference count will be decremented if it
    is above zero. When an initialized display's reference count reaches
    zero, termination will occur.

Replace the second sentence of the last paragraph with:

    All displays start out uninitialized with a reference count of zero.

Add to the end of section "3.3 EGL Queries".

   To query non-string attributes of an initialized display, use:

        EGLBoolean eglQueryDisplayAttribKHR(EGLDisplay dpy,
                                            EGLint name,
                                            EGLAttrib *value);

    On success, EGL_TRUE is returned, and the value of the attribute
    specified by <name> is returned in the space pointed to by <value>.

    On failure, EGL_FALSE is returned.  An EGL_NOT_INITIALIZED error
    is generated if EGL is not initialized for <dpy>.  An
    EGL_BAD_ATTRIBUTE error is generated if <name> is not a valid
    value. Currently, the only valid attribute name is
    EGL_TRACK_REFERENCES_KHR.

Interactions with EGL_KHR_platform_android:

    If eglGetPlatformDisplay() is called with <platform> set to
    EGL_PLATFORM_ANDROID_KHR, the default value of
    EGL_TRACK_REFERENCES_KHR is EGL_TRUE.

Interactions with EGL_EXT_platform_device, EGL_KHR_platform_gbm,
EGL_KHR_platform_x11, and EGL_KHR_platform_wayland:

    If eglGetPlatformDisplay() is called with <platform> set to
    EGL_PLATFORM_DEVICE_EXT, EGL_PLATFORM_GBM_KHR, EGL_PLATFORM_X11_KHR,
    or EGL_PLATFORM_WAYLAND_KHR, the default value of
    EGL_TRACK_REFERENCES_KHR is EGL_FALSE.

Interactions with EGL_EXT_device_query:

    The eglQueryDisplayAttribKHR function defined here is equivalent to
    eglQueryDisplayAttribEXT defined by EGL_EXT_device_query, and the
    attribute names supported are a superset of those provided by both
    extensions and any others which rely on them.

Issues

    1.  What is the default value for EGL_TRACK_REFERENCES_KHR?

        RESOLUTION: For backwards compatibility reasons, the default
        value is platform-specific. The Android platform has
        historically implemented the behavior of
        EGL_TRACK_REFERENCES_KHR = EGL_TRUE, while other platforms
        defaulted to the opposite behavior. Application components
        capable of supporting either behavior will be able to query
        the value to determine how to proceed.

    2.  Should the value of EGL_TRACK_REFERENCES_KHR affect whether
        eglGetPlatformDisplay returns a new display handle or an
        existing one given otherwise identical parameters?

        RESOLUTION: Yes. For any given combination of platform display
        handle and other attributes, calling eglGetPlatformDisplay
        with different values for EGL_TRACK_REFERENCES_KHR will result
        in two different EGLDisplay handles being returned.

        Resources created with respect to one of these EGLDisplays will
        not be accessible to the other. This restriction is unlikely to
        cause issues, because the reference counting is added primarily
        to support independent toolkits. Application components which
        independently initialize and terminate the display are not
        likely to share resources, particularly if they use different
        methods for that initialization.

    3.  Should the new display attribute be queryable?

        RESOLUTION: Yes. Not all implemenations will support both TRUE
        and FALSE for this attribute. Application components capable of
        supporting either value will allow the default to be chosen, and
        then query the value to determine how to handle termination.

    4.  Should implementations which support this extension be required
        to support both TRUE and FALSE for the attribute?

        RESOLUTION: No. Lack of refcounting in the core specification is
        considered by many to be a flaw, and some implementations/platforms
        will choose to always provide refcounting behavior. This technically
        makes them non-compliant. The addition of this extension should allow
        that deviation.

Revision History

    #4 (March 15, 2018) Jon Leech

        - Change extension number from 118 to 126 to avoid an accidental
          collision.

    #3 (January 12, 2017) Daniel Kartch

        - Change to KHR.
        - Allocate enum value.

    #2 (November 15, 2016) Daniel Kartch

        - Full termination portion split off into separate extension
          EGL_XXX_full_termination.
        - Update reference counting to have separate EGLDisplays for
          the same native display, one with reference counting and
          one without.
        - Add query function to determine attribute value.

    #1 (October 28, 2014) James Jones

        - Initial draft as EGL_XXX_display_reference
