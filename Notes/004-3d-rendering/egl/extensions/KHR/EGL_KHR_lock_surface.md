# KHR_lock_surface

Name

    KHR_lock_surface

Name Strings

    EGL_KHR_lock_surface

Contributors

    Gary King
    Jon Leech
    Marko Lukat
    Tim Renouf

Contacts

    Jon Leech (jon 'at' alumni.caltech.edu)

Notice

    Copyright (c) 2006-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    This extension, as well as the layered EGL_KHR_lock_surface2, are
    obsolete and have been replaced by EGL_KHR_lock_surface3. Khronos
    recommends implementers who support this extension and lock_surface2
    also implement lock_surface3, and begin transitioning developers to
    using that extension. See issue 17 for the reason.

    Complete.
    Version 17 approved by the Khronos Board of Promoters on
    February 11, 2008.

Version

    Version 19, October 15, 2013

Number

    EGL Extension #2

Dependencies

    Requires EGL 1.0

    This extension is written against the wording of the EGL 1.3
    Specification.

Overview

    This extension allows mapping color buffers of EGL surfaces into the
    client address space. This is useful primarily for software
    rendering on low-end devices which do not support EGL client
    rendering APIs, although it may be implemented efficiently on more
    capable devices as well.

    There is a newer EGL_KHR_lock_surface2 extension which slightly
    modifies and clarifies the semantics of this extension. Vendors
    should refer to EGL_KHR_lock_surface2 before deciding to implement
    only EGL_KHR_lock_surface.

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
    of eglChooseConfig:

        EGL_FORMAT_RGB_565_EXACT_KHR        0x30C0
        EGL_FORMAT_RGB_565_KHR              0x30C1
        EGL_FORMAT_RGBA_8888_EXACT_KHR      0x30C2
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

       "If EGL_LOCK_SURFACE_BIT_KHR is set in EGL_SURFACE_TYPE_KHR, then
        a surface created from the EGLConfig may be locked, mapped into
        client memory, and unlocked. Locking is described in section
        3.5.6. If EGL_OPTIMAL_FORMAT_BIT_KHR is set in
        EGL_SURFACE_TYPE_KHR, then the surface is considered optimal (by
        the implementation) from a performance standpoint when buffer
        mapping is being done.

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


    Add a new section following the current section 3.5.5:

       "3.5.6 Locking and Mapping Rendering Surfaces

        A rendering surface may be <locked> by calling

            EGLBoolean eglLockSurfaceKHR(EGLDisplay dpy,
                                         EGLSurface surface,
                                         const EGLint *attrib_list);

        While a surface is locked, only two operations can be performed
        on it. First, the color buffer of the surface may be <mapped>,
        giving a pointer into client memory corresponding to the memory
        of the mapped buffer, and attributes describing mapped buffers
        may be queried. Second, the surface may be unlocked. Any
        attempts to use a locked surface in other EGL APIs will fail and
        generate an EGL_BAD_ACCESS error.

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
        refers to the back buffer

           [fn: "mapped" only means that the pointer returned is
            intended to *correspond* to graphics memory. Implementation
            are not required to return an actual pointer into graphics
            memory, and often will not.]

        The contents of the mapped buffer are initially undefined(fn)
        unless the EGL_MAP_PRESERVE_PIXELS_KHR attribute of
        eglLockSurfaceKHR is set to EGL_TRUE, in which case the contents
        of the buffer are taken from the contents of <surface>'s color
        buffer. The default value of EGL_MAP_PRESERVE_PIXELS_KHR is
        EGL_FALSE.

           [fn: In order to avoid pipeline stalls and readback delays on
            accelerated implementations, we do not mandate that the
            current contents of a color buffer appear when it's mapped
            to client memory, unless the EGL_MAP_PRESERVE_PIXELS_KHR
            flag is set. Applications using mapped buffers which are not
            preserved must write to every pixel of the buffer before
            unlocking the surface. This constraint is considered
            acceptable for the intended usage scenario (full-frame
            software renderers).]

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
        origin, pixel size, and the bit width and location of each color
        component within a pixel. These attributes may be queried using
        eglQuerySurface, and are described in more detail in section
        3.5.7.

        The EGL_BITMAP_POINTER_KHR and EGL_BITMAP_PITCH_KHR attributes
        of a locked surface may change following successive calls to
        eglLockSurfaceKHR(fn), so they must be queried each time a
        buffer is mapped. Other attributes of a mapped buffer are
        invariant and need be queried only once following surface
        creation.

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
            within a 16- (for RGB565 formats) or 32-bit (for RGBA8888
            formats) integer containing the pixel(fn). If a color
            component does not exist in the mapped buffer, then the bit
            offset of that component is zero.

        In addition to these attributes, the number of bits for each
        color component of a pixel within a mapped buffer is obtained by
        querying the EGL_<x>_SIZE attribute of the EGLConfig used to
        create the surface, where <x> is <x> is one of RED, GREEN, BLUE,
        ALPHA, or LUMINANCE. The size of a pixel in the mapped buffer,
        in bytes, can be determined by querying the EGL_BUFFER_SIZE
        attribute of the EGLConfig, rounding up to the nearest multiple
        of 8, and converting from bits to bytes.

        Querying EGL_BITMAP_POINTER_KHR and EGL_BITMAP_PITCH_KHR is only
        allowed when <surface> is mapped (see section 3.5.6). Querying
        either of these attributes for the first time after calling
        eglLockSurfaceKHR causes the color buffer of the locked surface
        to be mapped. Querying them again before unlocking the surface
        will return the same values as the first time. However, after
        calling eglUnlockSurfaceKHR, these properties become undefined.
        After a second call to eglLockSurfaceKHR, these properties may
        again be queried, but their values may have changed.

        Other properties of the mapped color buffer of a surface are
        invariant, and need be queried only once following surface
        creation. If <surface> was created with an EGLConfig whose
        EGL_SURFACE_TYPE attribute does not contain
        EGL_LOCK_SURFACE_BIT_KHR, queries of EGL_BITMAP_ORIGIN_KHR and
        EGL_BITMAP_PIXEL_<x>_OFFSET_KHR return undefined values."

    Add to the description of eglQuerySurface errors in the last
    paragraph of section 3.5.6 (renumbered to 3.5.7) on page 34:

       "... If <attribute> is either EGL_BITMAP_POINTER_KHR or
        EGL_BITMAP_PITCH_KHR, and either <surface> is not locked using
        eglLockSurfaceKHR, or <surface> is locked but mapping fails,
        then an EGL_BAD_ACCESS error is generated."

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

    RESOLVED: Not initially. However, the <attrib_list> parameter of
    eglLockSurfaceKHR supports this in the future. There is no <buffer>
    attribute to eglQuerySurface, so such a layered extension would have
    to either create a new naming convention (such as
    EGL_BITMAP_{DEPTH,COLOR,STENCIL,ALPHA_MASK}_POINTER), or define an
    extended query eglQuerySurfaceBuffer() which takes a <buffer>
    parameter. It would also be tricky to support interleaved depth /
    stencil formats. But the attribute list offers some future-proofing
    at low cost.

 4) What properties of mapped buffers can be queried?

    RESOLVED: A pointer to the buffer and its pitch, both of which may
    change in successive lock/unlock cycles. These may be queried only
    while the underlying surface is locked, and are undefined after
    unlocking. The first query following locking is the point at which
    actual buffer mapping must occur.

    RESOLVED: Additionally, the pixel size, origin, and color component
    bitfield size and offset for each component, which are invariant
    and may be queried at any time.

 5) How are mode changes indicated? What happens to the mapped
    buffer during a mode change?

    RESOLVED: UnlockSurfaceKHR fails and raises an error if a mode
    change occurred while the surface was locked (although the surface
    still ends up in the unlocked state - this is necessary since
    there's no way to clear the error!). If a mode change occurs while a
    buffer is mapped, the implementation must still allow the
    application to access mapped buffer memory, even though the contents
    of the mapped buffer may not be reflected in the changed window
    after unmapping.

    Note: There's no convenient way to indicate mode changes while
    a surface is unlocked, despite that being useful to tell apps they
    have to redraw. The problem is that we don't have an event system,
    and the power management functionality is overkill since the only
    resources which are likely to be damaged by a mode change are
    visible window contents. Fortunately, this problem is beyond the
    scope of this extension.

 6) Does locking a surface imply mapping its buffers?

    RESOLVED: No. Locking simply places the surface in that state and
    prevents it from being made current / swapped / etc. Buffers are
    mapped only when their pointers or pitch are queried using
    eglQuerySurface.

    An interesting side effect of this resolution is that calling
    eglLockSurfaceKHR immediately followed by eglUnlockSurfaceKHR DOES
    NOT CHANGE THE CONTENTS OF BUFFERS, since none of them were mapped.
    Likewise locking a surface, querying a buffer pointer or pitch, and
    then unlocking it without changing the mapped buffer contents causes
    the surface contents of the mapper buffer(s) to become undefined.

    At the Atlanta F2F, there was a suggestion that eglLockSurfaceKHR
    should immediately map the color buffer and return a pointer to it,
    on the basis that this would make it harder for applications to
    mistakenly use an old buffer pointer from a previous mapping cycle.
    At the same time, people working on more powerful GPUs wanted the
    lock operation to be lightweight. These are not consistent goals and
    we have thus far chosen to separate the lightweight locking, and
    more expensive mapping operations.

 7) Can buffer contents be preserved in mapping?

    RESOLVED: Yes. The default behavior is to discard / leave undefined
    the mapped buffer contents, but the EGL_MAP_PRESERVE_PIXELS_KHR flag
    may be specified to eglLockSurfaceKHR.

 8) Should usage hints be provided during mapping?

    RESOLVED: Yes, they may be provided in the EGL_LOCK_USAGE_HINT_KHR
    bitmask attribute to eglLockSurfaceKHR. Implementations are required
    to behave correctly no matter the value of the flag vs. the
    operations actually performed, so the hint may be ignored.

 9) Should we be able to lock subrectangles of a surface?

    RESOLVED: No. However, the attribute list parameter of
    eglLockSurfaceKHR allows a layered extension to implement this
    behavior by specifying an origin and size to map within the buffer.

10) Should the BITMAP_PIXEL_<component>_OFFSET attributes belong to the
    surface, or the config?

    RESOLVED: Part of the surface. Configs supporting a specific format
    are matched with config attribute EGL_MATCH_FORMAT_KHR, which
    supports specific bit-exact formats such as
    EGL_FORMAT_565_EXACT_KHR.

11) Can the pixel size in a mapped buffer be derived from the
    EGL_BUFFER_SIZE attribute of the config used to create it?

    RESOLVED: Yes. In principle, hardware using padding bytes in its
    framebuffer storage could be a problem, and a separate
    BITMAP_PIXEL_SIZE surface attribute would be needed. However, we
    don't think implementations are likely to waste graphics memory and
    bandwidth in this fashion.

12) How are color component locations within a pixel described?

    RESOLVED: Each R, G, B, and A component has a queryable bit offset
    within an integer. The size of the integer depends on the total size
    of the pixel; for the 565 formats, the pixel is a 16-bit integer.
    For the 8888 formats, the pixel is a 32-bit integer.

    We cannot describe component locations with byte locations, since
    the 565 formats have components straddling byte boundaries. However,
    this means that offsets for the RGBA8888_EXACT format are different
    between little- and big-endian CPUs, since the desired format is B,
    G, R, A components laid out as bytes in increasing memory order.

13) Can mapped buffer contents be affected by other EGL operations?

    RESOLVED: No. A locked surface only allows two operations:
    unlocking, and mapping. No other EGL operations can take place while
    the surface is locked (if this were not the case, then
    eglSwapBuffers might destroy the contents of a mapped buffer).

    It is possible that operations outside the scope of EGL could affect
    a mapped color buffer. For example, if a surface's color buffer were
    made up of an EGLImage, one of the EGL client APIs could draw to
    that image while it was mapped. Responsibility for avoiding this
    situation is in the hands of the client.

14) Can EGL_MATCH_FORMAT_KHR be queried for a config?

    RESOLVED: Yes. Unlockable configs return EGL_NONE for this
    attribute.

15) Is a goal of this extension to support "mixed-mode" rendering (both
    software and EGL client API rendering to the same surface)?

    RESOLVED: No. An implementation *can* choose to export configs
    supporting creation of lockable surfaces which also support
    rendering by OpenGL ES, OpenVG, or other client APIs (when the
    surface is not locked). But there is nothing in the extension
    requiring this, and the motivation for the extension is simply to
    support software rendering.

16) Can mapping a locked surface fail?

    RESOLVED: Yes, if memory can't be allocated in the client. This is
    indicated by queries of EGL_BITMAP_POINTER_KHR and
    EGL_BITMAP_PITCH_KHR failing and generating an EGL_BAD_ACCESS error.

17) Why has this extension been obsoleted and replaced by
    EGL_KHR_lock_surface3?

    RESOLVED: Starting with the December 4, 2013 release of EGL 1.4, EGLint
    is defined to be the same size as the native platform "int" type. Handle
    and pointer attribute values *cannot* be represented in attribute lists
    on platforms where sizeof(handle/pointer) > sizeof(int). Existing
    extensions which assume this functionality are being replaced with new
    extensions specifying new entry points to work around this issue. See
    the latest EGL 1.4 Specification for more details.

Revision History

    Version 19, 2013/10/15 - Add issue 17 explaining that the bitmap pointer
        cannot be safely queried using this extension on 64-bit platforms,
        and suggest EGL_KHR_lock_surface3 instead. Change formal parameter
        names from 'display' to 'dpy' to match other EGL APIs.
    Version 18, 2010/03/23 - Added introductory remark referring to the
        EGL_KHR_lock_surface2 extension. Clarified that it is the back
        buffer of a back-buffered surface that is mapped.
    Version 17, 2008/10/08 - Updated status (approved as part of
        OpenKODE 1.0).
    Version 16, 2008/01/24 - Add issue 16 noting that mapping can fail,
        and a corresponding new error condition for eglQuerySurface.
        Clean up the issues list.
    Version 15, 2008/01/09 - Add issue 15 noting that supporting
        mixed-mode rendering is not a goal or requirement of the
        extension.
    Version 14, 2007/11/07 - change ARGB_8888_EXACT back to
        RGBA_8888_EXACT, since the offsets are now dependent on the
        endianness of the CPU. Add issue 12 describing this, and clarify
        that offsets are within a 16- or 32-bit integer depending on the
        format. Added issue 13 clarifying that locked buffer contents
        are not affected by eglSwapBuffers, because eglSwapBuffers
        cannot be issued on a mapped surface. Allow querying
        EGL_MATCH_FORMAT_KHR for a config, and added related issue 14.
    Version 13, 2007/05/10 - change RGBA_8888_EXACT to ARGB_8888_EXACT
        to match hardware layout.
    Version 12, 2007/04/06 - clarify that when EGL_MATCH_FORMAT_KHR is
        EGL_DONT_CARE, it does not affect component size of selected
        configs.
    Version 11, 2007/04/05 - add missing KHR suffix to some tokens.
    Version 10, 2007/04/05 - assign enumerant values. Add OpenKODE 1.0
        Provisional disclaimer.
    Version 9, 2007/03/26 - add format tokens to "New Tokens"
        section. Correct description of RGBA format tokens.
    Version 8, 2007/03/26 - add issue 11 noting theo