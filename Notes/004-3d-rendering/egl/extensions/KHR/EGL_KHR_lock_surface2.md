# KHR_lock_surface2

Name

    KHR_lock_surface2

Name Strings

    EGL_KHR_lock_surface2

Contributors

    Mark Callow
    Gary King
    Jon Leech
    Marko Lukat
    Alon Or-bach
    Tim Renouf

Contacts

    Jon Leech (jon 'at' alumni.caltech.edu)

Notice

    Copyright (c) 2006-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    This extension is obsolete and has been replaced by
    EGL_KHR_lock_surface3. Khronos recommends implementers who support this
    extension also implement lock_surface3, and begin transitioning
    developers to using that extension. See issue 21 for the reason.

    Complete. Version 2 approved by the Khronos Board of Promoters on
    May 28, 2010.
    Implemented by Antix Labs.

Version

    Version 3, December 4, 2013

Number

    EGL Extension #16

Dependencies

    Requires EGL 1.0 and EGL_KHR_lock_surface version 18.

    This extension is written against the wording of the EGL 1.3
    and EGL 1.4 Specifications. Unless otherwise specified, each change
    applies to both specifications. Unless otherwise specified, a page
    number refers to the EGL 1.3 specification.

    This extension is written against the wording of EGL_KHR_lock_surface
    version 18.

Overview

    This extension slightly modifies and clarifies some semantic aspects
    of the EGL_KHR_lock_surface extension, in a way that is backwards
    compatible for applications.

    The extension is presented here as the full text of the
    EGL_KHR_lock_surface extension (minus the Status, Version, Number and
    Dependencies sections at the start) as modified by the changes made by
    this EGL_KHR_lock_surface2 extension. A diff utility can be used between
    EGL_KHR_lock_surface version 18 and this EGL_KHR_lock_surface2 extension
    to show the exact changes.

    An application which needs to tell whether the implementation supports
    EGL_KHR_lock_surface2, or just the original EGL_KHR_lock_surface, can
    use eglQueryString with EGL_EXTENSIONS to query the list of
    implemented extensions.

    The changes over EGL_KHR_lock_surface can be summarized as follows:

      * EGL_KHR_lock_surface had the EGL_MAP_PRESERVE_PIXELS_KHR attribute on
        eglLockSurfaceKHR, but failed to point out how the surface attribute
        EGL_SWAP_BEHAVIOR would interact with lock surface rendering.
        EGL_KHR_lock_surface2 specifies that the locked buffer contains the
        back buffer pixels if EGL_SWAP_BEHAVIOR is EGL_BUFFER_PRESERVED
        _or_ if EGL_MAP_PRESERVE_PIXELS_KHR is EGL_TRUE, and provides a way to
        set EGL_SWAP_BEHAVIOR on creation of a lockable window surface,
        even if EGL_SWAP_BEHAVIOR is not otherwise modifiable.
        EGL_SWAP_BEHAVIOR now defaults to EGL_BUFFER_PRESERVED for a
        lockable surface.

      * EGL_KHR_lock_surface failed to specify its interaction with the
        EGL requirement that a context be current at eglSwapBuffers; no
        context is used for lock surface rendering. EGL_KHR_lock_surface2
        relaxes that requirement for a lockable window surface, in a way
        that is anticipated to apply to all window surfaces in a future
        version of EGL.

      * Wording in EGL_KHR_lock_surface could be read to imply that almost
        all surface attributes are invariant for a lockable surface.
        EGL_KHR_lock_surface2 clarifies the wording.

      * EGL_KHR_lock_surface2 clarifies what is returned when
        the attribute EGL_MATCH_FORMAT_KHR is queried, especially when
        one of the "inexact" formats was used to choose the config.

      * EGL_KHR_lock_surface did not specify when a surface could change size.
        EGL_KHR_lock_surface2 specifies that a surface cannot change size
        when it is locked.

      * EGL_KHR_lock_surface2 adds the config attribute
        EGL_BITMAP_PIXEL_SIZE_KHR, to allow an application to dynamically
        detect pixel layout for a format with a "hole", such as RGBU8888
        (where "U" means "unused").

New Tokens

    Accepted by the <attribute> parameter of eglQuerySurface:

        EGL_BITMAP_PIXEL_SIZE_KHR           0x3110

Full text of EGL_KHR_lock_surface plus EGL_KHR_lock_surface2:

Overview

    This extension allows mapping color buffers of EGL surfaces into the
    client address space. This is useful primarily for software
    rendering on low-end devices which do not support EGL client
    rendering APIs, although it may be implemented efficiently on more
    capable devices as well.

New Types

    None

New Procedures and Functions

    EGLBoolean eglLockSurfaceKHR(EGLDisplay dpy, EGLSurface surface,
                                 const EGLint *attrib_list);
    EGLBoolean eglUnlockSurfaceKHR(EGLDisplay dpy,
                                   EGLSurface surface);

New Tokens

    Returned in the EGL_SURFACE_TYPE bitmask attribute of EGLConfigs:

        EGL_LOCK_SURFACE_BIT_KHR            0x0080
        EGL_OPTIMAL_FORMAT_BIT_KHR          0x0100

    Accepted as an attribute name in the <attrib_list> argument of
    eglChooseConfig, and the <attribute> argument of eglGetConfigAttrib:

        EGL_MATCH_FORMAT_KHR                0x3043

    Accepted as attribute values for the EGL_MATCH_FORMAT_KHR attribute
    of eglChooseConfig, and defined as possible values of that attribute
    when querying it:

        EGL_FORMAT_RGB_565_EXACT_KHR        0x30C0
        EGL_FORMAT_RGBA_8888_EXACT_KHR      0x30C2

    Accepted as attribute values for the EGL_MATCH_FORMAT_KHR attribute
    of eglChooseConfig:

        EGL_FORMAT_RGB_565_KHR              0x30C1
        EGL_FORMAT_RGBA_8888_KHR            0x30C3

    Accepted as attribute names in the <attrib_list> argument of
    eglLockSurfaceKHR:

        EGL_MAP_PRESERVE_PIXELS_KHR         0x30C4
        EGL_LOCK_USAGE_HINT_KHR             0x30C5

    Accepted as bit values in the EGL_LOCK_USAGE_HINT_KHR bitmask attribute
    of eglLockSurfaceKHR:

        EGL_READ_SURFACE_BIT_KHR            0x0001
        EGL_WRITE_SURFACE_BIT_KHR           0x0002

    Accepted by the <attribute> parameter of eglQuerySurface:

        EGL_BITMAP_POINTER_KHR              0x30C6
        EGL_BITMAP_PITCH_KHR                0x30C7
        EGL_BITMAP_ORIGIN_KHR               0x30C8
        EGL_BITMAP_PIXEL_RED_OFFSET_KHR     0x30C9
        EGL_BITMAP_PIXEL_GREEN_OFFSET_KHR   0x30CA
        EGL_BITMAP_PIXEL_BLUE_OFFSET_KHR    0x30CB
        EGL_BITMAP_PIXEL_ALPHA_OFFSET_KHR   0x30CC
        EGL_BITMAP_PIXEL_LUMINANCE_OFFSET_KHR 0x30CD
        EGL_BITMAP_PIXEL_SIZE_KHR           0x3110

    Returns in the *<value> parameter of eglQuerySurface when
    <attribute> is EGL_BITMAP_ORIGIN_KHR:

        EGL_LOWER_LEFT_KHR                  0x30CE
        EGL_UPPER_LEFT_KHR                  0x30CF

Additions to Chapter 2 of the EGL 1.3 Specification (EGL Operation)

    Add to the end of section 2.2.2:

        Finally, some surfaces may be <locked>, which allows the
        implementation to map buffers of that surface into client memory
        for use by software renderers(fn). Locked surfaces cannot be
        used for any other purpose. When a locked surface is <unlocked>,
        any changes to the mapped buffer(s) are reflected in the actual
        graphics or system memory containing the surface.

           [fn: on implementations not supporting mapping graphics
            memory, or which do not wish to take the stability and
            security risks that entail, mapping may be done using
            copy-out and copy-in behavior.]

Additions to Chapter 3 of the EGL 1.3 Specification (EGL Functions and Errors)

    Add to the description of the EGL_BAD_ACCESS error in section 3.1:

       "... or, a surface is locked)."

    Add to table 3.2 ("Types of surfaces supported by an EGLConfig")

        EGL Token Name              Description
        --------------------        ------------------------------------
        EGL_LOCK_SURFACE_BIT_KHR    EGLConfig allows locking surfaces
        EGL_OPTIMAL_FORMAT_BIT_KHR  This format is considered optimal
                                    (preferred) when locking / mapping /
                                    unlocking is being done.

    Change the first paragraph under "Other EGLConfig Attribute
    Descriptions" on p. 16:

       "EGL_SURFACE_TYPE is a mask indicating both the surface types
        that can be created by the corresponding EGLConfig (the config
        is said to <support> those surface types), and the optional
        behaviors such surfaces may allow. The valid bit settings are
        shown in Table 3.2."

    Add a new paragraph following the second paragraph of the same
    section:

       "If EGL_LOCK_SURFACE_BIT_KHR is set in EGL_SURFACE_TYPE, then
        a surface created from the EGLConfig may be locked, mapped into
        client memory, and unlocked. Locking is described in section
        3.5.6. If EGL_OPTIMAL_FORMAT_BIT_KHR is set in
        EGL_SURFACE_TYPE, then the surface is considered optimal (by
        the implementation) from a performance standpoint when buffer
        mapping is being done."

    Replace the second paragraph of section 3.3 "EGL Versioning":

       "The EGL_CLIENT_APIS string describes which client rendering APIs
        are supported. It is zero-terminated and contains a
        space-separated list of API names, which may include
        ``OpenGL_ES'' if OpenGL ES is supported, and ``OpenVG'' if
        OpenVG is supported. If no client APIs are supported, then the
        empty string is returned."

    Insert a new paragraph and table in section 3.4.1 "Querying
    Configurations", following the description of
    EGL_MATCH_NATIVE_PIXMAP on page 21:

       "If EGL_MATCH_FORMAT_KHR is specified in <attrib_list>, it must
        be followed by one of the attribute values EGL_DONT_CARE,
        EGL_NONE, or one of the format tokens in table
        [locksurf.format].

        When EGL_MATCH_FORMAT_KHR has the value EGL_NONE, only configs
        which cannot be locked or mapped will match. Such configs must
        not have the EGL_LOCK_SURFACE_BIT_KHR set in EGL_SURFACE_TYPE.

        When EGL_MATCH_FORMAT_KHR has the value EGL_DONT_CARE, it is
        ignored.

        When EGL_MATCH_FORMAT_KHR has one of the values in table
        [locksurf.format], only EGLConfigs describing surfaces whose
        color buffers have the specified format, when mapped with
        eglLockSurface, will match this attribute. In this case, the
        EGL_<component>_SIZE attributes of resulting configs must agree
        with the specific component sizes specified by the format."

        Specific Format Name            Description
        --------------------            -----------
        EGL_FORMAT_RGB_565_EXACT_KHR    RGB565 fields in order from MSB to LSB within a 16-bit integer
        EGL_FORMAT_RGB_565_KHR          RGB565 fields in implementation-chosen order within a 16-bit integer
        EGL_FORMAT_RGBA_8888_EXACT_KHR  RGBA8888 fields in B, G, R, A byte order in memory
        EGL_FORMAT_RGBA_8888_KHR        RGBA8888 fields in implementation-chosen order within a 32-bit integer
        ------------------------------------------------------------------------------------------------------
        Table [locksurf.format]: Specific formats for mapped pixels.

    Add to table 3.4 ("Default values and match critera for EGLConfig
    attributes") on page 22:

        Attribute            Default         Selection   Sort    Sort
                                             Criteria    Order   Priority
        -------------------- -------------   ---------   -----   --------
        EGL_MATCH_FORMAT_KHR EGL_DONT_CARE   Exact       None    -

    Add EGL_MATCH_FORMAT_KHR to the last paragraph in section 3.4.1 on
    p. 23, describing attributes not used for sorting EGLConfigs.

    Add a new paragraph to the end of section 3.4.3 "Querying Configuration
    Attributes":

       "Querying the EGL_MATCH_FORMAT_KHR attribute results in EGL_NONE
        for an EGLConfig that is not lockable, one of the "exact" formats
        (EGL_FORMAT_RGB_565_EXACT_KHR, EGL_FORMAT_RGBA_8888_EXACT_KHR)
        if the color buffer matches that format when mapped with
        eglLockSurface, or for any other format a value that is not
        EGL_NONE or EGL_DONT_CARE but is otherwise undefined. In particular,
        the color buffer format matching one of the "inexact" formats
        does not guarantee that that EGL_FORMAT_* value is returned."

    In section 3.5.1 "Creating On-Screen Rendering Surfaces", add the
    following to the paragraph that lists the attributes that can be set
    in attrib_list:

       "... and EGL_SWAP_BEHAVIOR."

    and add a new penultimate paragraph:

       "EGL_SWAP_BEHAVIOR specifies the initial value of the
        EGL_SWAP_BEHAVIOR surface attribute (section 3.5.6), and is thus
        either EGL_BUFFER_PRESERVED or EGL_BUFFER_DESTROYED. This setting
        of EGL_SWAP_BEHAVIOR at surface creation time is supported only
        for a lockable surface, i.e. where the EGLConfig has
        EGL_LOCK_SURFACE_BIT_KHR set in EGL_SURFACE_TYPE."

    In EGL 1.4, also add the following text to that same paragraph:

       "For such a lockable surface, whether it is possible to change
        the EGL_SWAP_BEHAVIOR attribute after surface creation is
        determined by EGL_SWAP_BEHAVIOR_PRESERVED_BIT in the
        EGL_SURFACE_TYPE EGLConfig attribute."

    Add a new section following the current section 3.5.5:

       "3.5.6 Locking and Mapping Rendering Surfaces

        A rendering surface may be <locked> by calling

            EGLBoolean eglLockSurfaceKHR(EGLDisplay dpy,
                                         EGLSurface surface,
                                         const EGLint *attrib_list);

        While a surface is locked, only two operations can be performed
        on it. First, a surface attribute may be queried using
        eglQuerySurface. This includes the case of querying
        EGL_BITMAP_POINTER_KHR, which causes the surface to be
        <mapped> (if not already mapped) and gives
        a pointer into client memory corresponding to the memory
        of the mapped buffer. Second, the surface may be unlocked. Any
        attempts to use a locked surface in other EGL APIs will fail and
        generate an EGL_BAD_ACCESS error.

        While a surface is locked, its dimensions (the values of the EGL_WIDTH
        and EGL_HEIGHT surface attributes) do not change. They may change
        at any other time, therefore an application must query these
        attributes <after> the call to eglLockSurfaceKHR to ensure that it has
        the correct size of the mapped buffer.

        <attrib_list> specifies additional parameters affecting the locking
        operation. The list has the same structure as described for
        eglChooseConfig. Attributes that may be defined are shown in table
        [locksurf.attr], together with their default values if not specified
        in <attrib_list>, and possible values which may be specified for
        them in <attrib_list>.

        Attribute Name              Type    Default Value              Possible Values
        -----------------------     ------- -------------              -------------------------
        EGL_MAP_PRESERVE_PIXELS_KHR boolean EGL_FALSE                  EGL_TRUE / EGL_FALSE
        EGL_LOCK_USAGE_HINT_KHR     bitmask EGL_READ_SURFACE_BIT_KHR | Any combination of
                                            EGL_WRITE_SURFACE_BIT_KHR  EGL_READ_SURFACE_BIT_KHR
                                                                       and EGL_WRITE_SURFACE_BIT_KHR
        --------------------------------------------------------------
        Table [locksurf.attr]: eglLockSurfaceKHR attribute names,
        default values, and possible values.

        On failure, the surface is unaffected and eglLockSurfaceKHR
        returns EGL_FALSE. An EGL_BAD_ACCESS error is generated if any
        of these condition, are true:

          * <surface> was created with an EGLConfig whose
            EGL_SURFACE_TYPE attribute does not contain
            EGL_LOCK_SURFACE_BIT_KHR.
          * <surface> is already locked.
          * Any client API is current to <surface>.

        An EGL_BAD_ATTRIBUTE error is generated if an attribute or
        attribute value not described in table [locksurf.attr] is
        specified.

        Mapping Buffers of a Locked Surface
        -----------------------------------

        The color buffer of a locked surface can be <mapped> by calling
        eglQuerySurface (see section 3.5.7) with <attribute>
        EGL_BITMAP_POINTER_KHR(fn). The query returns a pointer to a
        buffer in client memory corresponding to the color buffer of
        <surface>. In the case of a back-buffered surface, color buffer
        refers to the back buffer.

           [fn: "mapped" only means that the pointer returned is
            intended to *correspond* to graphics memory. Implementation
            are not required to return an actual pointer into graphics
            memory, and often will not.]


        The contents of the mapped buffer are initially undefined(fn)
        unless either the EGL_MAP_PRESERVE_PIXELS_KHR attribute of
        eglLockSurfaceKHR is set to EGL_TRUE, or (for a window surface)
        the EGL_SWAP_BEHAVIOR surface attribute is set to
        EGL_BUFFER_PRESERVE, in which case the contents
        of the buffer are taken from the contents of <surface>'s color
        buffer. The default value of EGL_MAP_PRESERVE_PIXELS_KHR is
        EGL_FALSE.

           [fn: In order to avoid pipeline stalls and readback delays on
            accelerated implementations, we do not mandate that the
            current contents of a color buffer appear when it's mapped
            to client memory, unless the EGL_MAP_PRESERVE_PIXELS_KHR
            flag is set or (for a window surface) EGL_SWAP_BEHAVIOR is
            set to EGL_BUFFER_PRESERVE. Applications using mapped
            buffers which are not
            preserved must write to every pixel of the buffer before
            unlocking the surface. This constraint is considered
            acceptable for the intended usage scenario (full-frame
            software renderers). Such an application may lock-render-unlock
            multiple times per frame (i.e. per eglSwapBuffers) by setting
            EGL_MAP_PRESERVE_PIXELS_KHR to EGL_TRUE for the second and
            subsequent locks.

            Note that EGL_SWAP_BEHAVIOR also controls whether the color
            buffer contents are preserved over a call to eglSwapBuffers.]

        The EGL_LOCK_USAGE_HINT_KHR attribute of eglLockSurfaceKHR is a
        bitmask describing the intended use of the mapped buffer. If the
        mask contains EGL_READ_SURFACE_BIT_KHR, data will be read from
        the mapped buffer. If the mask contains
        EGL_WRITE_SURFACE_BIT_KHR, data will be written to the mapped
        buffer. Implementations must support both reading and writing to
        a mapped buffer regardless of the value of
        EGL_LOCK_USAGE_HINT_KHR, but performance may be better if the
        hint is consistent with the actual usage of the buffer. The
        default value of EGL_LOCK_USAGE_HINT_KHR hints that both reads
        and writes to the mapped buffer will be done.

        Other attributes of the mapped buffer describe the format of
        pixels it contains, including its pitch (EGL_BITMAP_PITCH_KHR),
        origin (EGL_BITMAP_ORIGIN_KHR), and the bit location of each color
        component within a pixel (EGL_BITMAP_PIXEL_x_OFFSET_KHR). These
        attributes may be queried using eglQuerySurface, and are described
        in more detail in section 3.5.7.

        The EGL_BITMAP_POINTER_KHR and EGL_BITMAP_PITCH_KHR attributes
        of a locked surface may change following successive calls to
        eglLockSurfaceKHR(fn), so they must be queried each time a
        buffer is mapped. Other attributes of a mapped buffer (listed in
        the paragraph above) are invariant and need be queried only once
        following surface creation.

           [fn: The pointer and pitch of a mapped buffer may change due
            to display mode changes, for example.]

        Mapping will not suceed if client memory to map the surface into
        cannot be allocated. In this case, querying eglQuerySurface with
        <attribute> EGL_BITMAP_POINTER_KHR will fail and generate an EGL
        error.

        Unlocking Surfaces
        ------------------

        A rendering surface may be <unlocked> by calling

            EGLBoolean eglUnlockSurfaceKHR(EGLDisplay dpy,
                                           EGLSurface surface);

        Any mapped buffers of <surface> become unmapped following
        eglUnlockSurfaceKHR. Any changes made to mapped buffers of
        <surface> which it was locked are reflected in the surface after
        unlocking(fn).

           [fn: This language enables implementations to either map
            video memory, or copy from a separate buffer in client
            memory.]

        If <surface> was created with an EGLConfig whose
        EGL_SURFACE_TYPE attribute contains EGL_OPTIMAL_FORMAT_BIT_KHR,
        then the surface is considered optimal (by the implementation)
        from a performance standpoint when buffer mapping is being
        done(fn).

           [fn: This often means that the format of all mapped buffers
            corresponds directly to the format of those buffers in
            <surface>, so no format conversions are required during
            unmapping. This results in a high-performance software
            rendering path. But "optimal format" is really just a hint
            from EGL that this config is preferred, whatever the actual
            reason.]

        On failure, eglUnlockSurfaceKHR returns EGL_FALSE. An
        EGL_BAD_ACCESS error is generated if any of these conditions are
        true:

          * <surface> is already unlocked.
          * A display mode change occurred while the surface was locked,
            and the implementation was unable to reflect mapped buffer
            state(fn). In this case, <surface> will still be unlocked.
            However, the contents of the previously mapped buffers of
            <surface> become undefined, rather than reflecting changes
            made in the mapped buffers in client memory.

           [fn: Usually this may only occur with window surfaces which
            have been mapped. EGL does not have an event mechanism to
            indicate display mode changes. If such a mechanism exists
            (using native platform events or the OpenKODE event system),
            applications should respond to mode changes by regenerating
            all visible window content, including re-doing any software
            rendering overlapping the mode change.]"

    Add to table 3.5 ("Queryable surface attributes and types")

        Attribute                   Type    Description
        ---------                   ----    -----------
        EGL_BITMAP_POINTER_KHR      pointer Address of a mapped color buffer (MCB).
        EGL_BITMAP_PITCH_KHR        integer Number of bytes between the start of
                                            adjacent rows in an MCB.
        EGL_BITMAP_ORIGIN_KHR       enum    Bitmap origin & direction
        EGL_BITMAP_PIXEL_x_OFFSET_KHR integer Bit location of each color buffer
                                              component within a pixel in an MCB.
        EGL_BITMAP_PIXEL_SIZE_KHR   integer Bits per pixel

    In EGL 1.4 only, in the description of eglSurfaceAttrib properties
    that can be set in section 3.5.6 (renumbered to 3.5.7), on page 35,
    add to the first paragraph describing EGL_SWAP_BEHAVIOR:

       "The value of EGL_SWAP_BEHAVIOR also affects the semantics of
        eglLockSurfaceKHR for a lockable window surface. See section 3.5.6."

    In EGL 1.4 only, in the description of eglSurfaceAttrib properties
    that can be set in section 3.5.6 (renumbered to 3.5.7), on page 35,
    change the paragraph concerning the initial value of EGL_SWAP_BEHAVIOR
    to:

       "The initial value of EGL_SWAP_BEHAVIOR is chosen by the
        implementation, except for a lockable window surface (i.e. where the
        EGLConfig has both EGL_LOCK_SURFACE_BIT_KHR and EGL_WINDOW_BIT set in
        EGL_SURFACE_TYPE), where the default is EGL_BUFFER_PRESERVED, but it
        may be overridden by specifying EGL_SWAP_BEHAVIOR to
        eglCreateWindowSurface."

    In EGL 1.3 only, in the description of eglQuerySurface properties
    that can be queried in section 3.5.6 (renumbered to 3.5.7), on page 33,
    add to the paragraph describing EGL_SWAP_BEHAVIOR:

       "The value of EGL_SWAP_BEHAVIOR also affects the semantics of
        eglLockSurfaceKHR for a lockable window surface. See section 3.5.6.
        For a lockable window surface (one whose EGLConfig has both
        EGL_LOCK_SURFACE_BIT_KHR and EGL_WINDOW_BIT set in EGL_SURFACE_TYPE),
        the value of this attribute may be set in the eglCreateWindowSurface
        call, and if not set there defaults to EGL_BUFFER_PRESERVED. See
        section 3.5.1. The default for a non-lockable surface is chosen by the
        implementation."

    Add to the description of eglQuerySurface properties in section
    3.5.6 (renumbered to 3.5.7) on page 33:

       "Properties of a bitmap surface which may be queried include:
          * EGL_BITMAP_POINTER_KHR, which maps the color buffer of a
            locked surface and returns the address in client memory of
            the mapped buffer.
          * EGL_BITMAP_PITCH_KHR, which returns the number of bytes
            between successive rows of a mapped buffer.
          * EGL_BITMAP_ORIGIN_KHR, which describes the way in which a
            mapped color buffer is displayed on the screen. Possible
            values are either EGL_LOWER_LEFT_KHR or EGL_UPPER_LEFT_KHR,
            indicating that the first pixel of the mapped buffer
            corresponds to the lower left or upper left of a visible
            window, respectively.
          * EGL_BITMAP_PIXEL_<x>_OFFSET_KHR, which describes the bit
            location of the least significant bit of each color
            component of a pixel within a mapped buffer. <x> is one of
            RED, GREEN, BLUE, ALPHA, or LUMINANCE.

            The offset for a color component should be treated as the
            number of bits to left shift the component value to place it
            within a n-bit
            integer containing the pixel, where n is the number of bits
            per pixel. If a color
            component does not exist in the mapped buffer, then the bit
            offset of that component is zero. If a color component
            does exist but not in a single contiguous range of bits,
            then the value of the attribute is EGL_UNKNOWN.

          * EGL_BITMAP_PIXEL_SIZE_KHR, which returns the number of bits
            per pixel, assumed to be least significant leftmost if there
            are multiple pixels per byte. The attribute takes the value
            EGL_UNKNOWN if this assumption is not true, or if pixels are not
            laid out left to right in memory (for example pairs of 16-bit
            pixels are swapped in memory).

        In addition to these attributes, the number of bits for each
        color component of a pixel within a mapped buffer is obtained by
        querying the EGL_<x>_SIZE attribute of the EGLConfig used to
        create the surface, where <x> is <x> is one of RED, GREEN, BLUE,
        ALPHA, or LUMINANCE.

        Querying EGL_BITMAP_POINTER_KHR and EGL_BITMAP_PITCH_KHR is only
        allowed when <surface> is mapped (see section 3.5.6). Querying
        either of these attributes for the first time after calling
        eglLockSurfaceKHR causes the color buffer of the locked surface
        to be mapped. Querying them again before unlocking the surface
        will return the same values as the first time. However, after
        calling eglUnlockSurfaceKHR, these properties become undefined.
        After a second call to eglLockSurfaceKHR, these properties may
        again be queried, but their values may have changed.

        Other properties of the mapped color buffer of a surface
        (in the list above) are
        invariant, and need be queried only once following surface
        creation. If <surface> was created with an EGLConfig whose
        EGL_SURFACE_TYPE attribute does not contain
        EGL_LOCK_SURFACE_BIT_KHR, queries of EGL_BITMAP_ORIGIN_KHR,
        EGL_BITMAP_PIXEL_<x>_OFFSET_KHR and EGL_BITMAP_PIXEL_SIZE_KHR
        return undefined values."

    Add to the description of eglQuerySurface errors in the last
    paragraph of section 3.5.6 (renumbered to 3.5.7) on page 34:

       "... If <attribute> is either EGL_BITMAP_POINTER_KHR or
        EGL_BITMAP_PITCH_KHR, and either <surface> is not locked using
        eglLockSurfaceKHR, or <surface> is locked but mapping fails,
        then an EGL_BAD_ACCESS error is generated."

    In section 3.9.3 Posting Semantics on page 46, append to the first
    paragraph:

       "This restriction does not apply to lockable surfaces; for such
        a surface, eglSwapBuffers and eglCopyBuffers may be called for
        a surface not bound to any client API context(fn).

           [fn: Normally this would only be done when using methods other
            than client API rendering to specify the color buffer contents,
            such as software rendering to a locked surface.]"

    and replace the second paragraph ("If <dpy> and <surface> ... not be
    executed until posting is completed.") with:

       "If <surface> is bound to a current client API context for the calling
        thread, eglSwapBuffers and eglCopyBuffers perform an implicit flush
        operation on the context (glFlush for an OpenGL or OpenGL ES context,
        vgFlush for an OpenVG context). Subsequent client API commands can be
        issued immediately, but will not be executed until posting is
        completed.

        If <surface> is current to a client API context in any thread other
        than the calling thread, eglSwapBuffers and eglCopyBuffers will fail.

    and append the following sentence to the eglSwapInterval paragraph:

       "The swap interval has no effect on an eglSwapBuffers for a surface
        not bound to a current client API context."

    In 3.9.4 Posting Errors, change the sentence "If <surface> is not bound
    to the calling thread's current context, an EGL_BAD_SURFACE error is
    generated." to:

       "If <surface> is bound to a current context in a thread other
        than the calling thread, an EGL_BAD_SURFACE error is generated."


Issues

 1) What is the rationale for this extension?

    Software renderers on low-end implementations need an efficient way
    to draw pixel data to the screen. High-end implementations must
    support the same interface for compatibility, while not compromising
    the accelerability of OpenGL ES and OpenVG client APIs using
    dedicated graphics hardware and memory.

    Using lock/unlock semantics enables low-end implementations to
    expose pointers directly into display memory (as extremely dangerous
    as that may be), while high-end implementations may choose to create
    backing store in client memory when mapping a buffer, and copy it to
    graphics memory when the surface is unlocked. Making the initial
    contents of a mapped buffer undefined means that no readbacks from
    graphics memory are required, avoiding pipeline stalls.

    This extension is not intended to support mixed-mode (client API and
    software) rendering. Since mapped buffer contents are undefined,
    unless the buffer is explicitly preserved (which may be unacceptably
    expensive on many implementations), applications doing software
    rendering must touch every pixel of mapped buffers at least once
    before unlocking the surface.

 2) Do we need to know if locked surfaces are "fast" or "native"?

    RESOLVED: Yes. This is indicated with the EGL_OPTIMAL_FORMAT_BIT_KHR
    of EGL_SURFACE_TYPE. However, note that there is no way to guarantee
    what "fast" or "no format conversions" really means; this is little
    more than an implementation hint.

 3) Should we be able to map buffers other than the color buffer?

    RESOLVED: Not initially. However, the <att