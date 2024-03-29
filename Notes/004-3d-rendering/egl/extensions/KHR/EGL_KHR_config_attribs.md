# KHR_config_attribs

Name

    KHR_config_attribs

Name Strings

    EGL_KHR_config_attribs

Contributors

    Jon Leech

Contacts

    Jon Leech (jon 'at' alumni.caltech.edu)

Notice

    Copyright (c) 2006-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Status

    Complete

Version

    Version 5, April 5, 2007

Number

    EGL Extension #1

Dependencies

    Requires EGL 1.2

    Some of the extended config attributes defined by this extension are
    only relevant when specific client APIs are supported.

    This extension is written against the wording of the EGL 1.2
    Specification. It exists for backwards compatibility with
    functionality introduced in EGL 1.3.

Overview

    This extension adds new EGL config attributes and attribute bits
    that express limitations of configs on a per-API basis, including
    whether client APIs created with respect to a config are expected to
    pass conformance, and which optional OpenVG color space and alpha
    mask format attributes are valid at surface creation time.

New Types

    None

New Procedures and Functions

    None

New Tokens

    New EGLConfig bitmask attribute name:

        EGL_CONFORMANT_KHR                  0x3042

    Valid bitfields in the EGL_SURFACE_TYPE bitmask attribute
    of EGLConfig:

        EGL_VG_COLORSPACE_LINEAR_BIT_KHR    0x0020
        EGL_VG_ALPHA_FORMAT_PRE_BIT_KHR     0x0040

Additions to Chapter 3 of the EGL 1.2 Specification (EGL Functions and Errors)

    Add to table 3.1, "EGLConfig attributes":

        Attribute           Type        Notes
        ---------           ----        -----
        EGL_CONFORMANT_KHR  bitmask     whether contexts created with
                                        this config are conformant

    Add to table 3.2, "Types of surfaces supported by an EGLConfig":

        EGL Token Name                  Description
        --------------                  -----------
        EGL_VG_COLORSPACE_LINEAR_BIT_KHR EGLConfig supports OpenVG rendering
                                        in linear colorspace
        EGL_VG_ALPHA_FORMAT_PRE_BIT_KHR EGLConfig supports OpenVG rendering
                                        with premultiplied alpha

    Add following the second paragraph of "Other EGLConfig Attribute
    Descriptions" in section 3.4 on p. 16:

       "If EGL_VG_COLORSPACE_LINEAR_BIT_KHR is set in EGL_SURFACE_TYPE,
        then the EGL_COLORSPACE attribute may be set to
        EGL_COLORSPACE_LINEAR when creating a window, pixmap, or pbuffer
        surface (see section 3.5)."

       "If EGL_VG_ALPHA_FORMAT_PRE_BIT_KHR is set in EGL_SURFACE_TYPE,
        then the EGL_ALPHA_FORMAT attribute may be set to
        EGL_ALPHA_FORMAT_PRE when creating a window, pixmap, or pbuffer
        surface (see section 3.5)."

    Add at the end of the fourth paragraph (description of
    EGL_CONFIG_CAVEAT) on p. 17:

       "... required OpenGL ES conformance tests (note that
        EGL_NON_CONFORMANT_CONFIG is obsolete, and the same information
        can be obtained from the EGL_CONFORMANT_KHR attribute on a
        per-client-API basis, not just for OpenGL ES."

       "EGL_CONFORMANT_KHR is a mask indicating if a client API context
        created with respect to the corresponding EGLConfig will pass
        the required conformance tests for that API. The valid bit
        settings are the same as for EGL_RENDERABLE_TYPE, as defined in
        table 3.3, but the presence or absence of each client API bit
        determines whether the corresponding context will be conformant
        or non-conformant(fn1)."

       "(fn1) most EGLConfigs should be conformant for all supported
        client APIs. Conformance requirements limit the number of
        non-conformant configs that an implementation can define."

    Add to the last paragraph of section 3.5.1 on p. 24 (describing
    eglCreateWindowSurface):

       "If <config> does not support the colorspace or alpha format
        attributes specified in <attrib_list> (e.g. if EGL_COLORSPACE is
        specified as EGL_COLORSPACE_LINEAR but the EGL_SURFACE_TYPE
        attribute of <config> does not include
        EGL_VG_COLORSPACE_LINEAR_BIT_KHR, or if EGL_ALPHA_FORMAT is
        specified as EGL_ALPHA_FORMAT_PRE but EGL_SURFACE_TYPE does not
        include EGL_VG_ALPHA_FORMAT_PRE_BIT_KHR), an EGL_BAD_MATCH error
        is generated."

    Add to the next-to-last paragraph of section 3.5.2 on p. 26
    (describing eglCreatePbufferSurface):

       "If <config> does not support the colorspace or alpha format
        attributes specified in <attrib_list> (as defined for
        eglCreateWindowSurface), an EGL_BAD_MATCH error is generated."

    Add to the last paragraph of section 3.5.4 on p. 29 (describing
    eglCreatePixmapSurface):

       "If <config> does not support the colorspace or alpha format
        attributes specified in <attrib_list> (as defined for
        eglCreateWindowSurface), an EGL_BAD_MATCH error is generated."

Issues

    1) How should colorspace and alpha format restrictions be specified?
       OpenVG implementations may not allow linear colorspace or
       premultiplied alpha rendering to all configs they support.

        RESOLVED: To maximize compatibility with EGL 1.3, we continue to
        specify the desired colorspace and alpha format at surface
        creation time. However, surface creation may fail if if the
        specified colorspace or alpha format are not supported.

        To allow apps to detect this situation, this extension adds
        EGLConfig attributes specifying *if* linear colorspace and/or
        premultiplied alpha formats are supported. If they are not
        supported, surface creation with the corresponding attributes
        set will fail with an EGL_BAD_MATCH error.

    2) How should the colorspace and alpha format capabilities be
       exposed in EGLConfigs?

        RESOLVED: as bitfields of the existing EGL_SURFACE_TYPE bitmask
        attribute.

        A separate bitmask might be more orthogonal, but there are
        plenty of unused bits in EGL_SURFACE_TYPE and this minimizes API
        and programming complexity.

    3) Are support for linear colorspace and and premultiplied alpha
       formats orthogonal?

        RESOLVED: Yes, according to the OpenVG Working Group. If they
        were not orthogonal, we could not specify them as independent
        bitfields.

    4) Should information about conformance be specified on a
       per-client-API basis?

        RESOLVED: Yes. This is needed for conformance testing and cannot
        be expressed by the EGL_CONFIG_CAVEAT attribute, which is OpenGL
        ES-specific.

    5) Should there also be a config attribute which specifies whether
       EGL_RENDER_BUFFER will be respected?

        UNRESOLVED: it would be consistent to add this attribute. but
        it's not clear if there's a requirement for doing so yet.

    6) Does this extension introduce a regression against EGL 1.2?

        RESOLVED: Yes. This is unavoidable, since we're allowing failure
        of surface creation that was required to succeed in the past.
        However, implementations that could not support the required
        colorspace or alpha mask format were effectively non-conformant
        (e.g. broken) in any event. The new EGL_SURFACE_TYPE attributes
        at least allow apps to know that their request will not be
        satisfied.

Dependencies on OpenGL ES

    If OpenGL ES is not supported, the EGL_OPENGL_ES_BIT in the
    EGL_CONFORMANT_KHR is irrelevant.

Dependencies on OpenVG

    If OpenVG is not supported, the EGL_OPENVG_BIT bit in
    EGL_CONFORMANT_KHR, and the EGL_VG_COLORSPACE_LINEAR_BIT_KHR and
    EGL_VG_ALPHA_FORMAT_PRE_BIT_KHR bits in EGL_SURFACE_TYPE, are
    irrelevant.

Revision History

    Version 5, 2007/04/05 - add enum values corresponding to EGL 1.3
        core features.
    Version 4, 2006/10/24 - prefix the bitfield names with "VG" to
        clarify that they only apply to OpenVG rendering to surfaces
        (although the corresponding core EGL_COLORSPACE and
        EGL_ALPHA_FORMAT attribute names do not currently include this
        prefix). Use "KHR" suffix instead of "OES".
    Version 3, 2006/10/15 - add new config attribute to express whether
        configs are conformant on a per-API basis. Correct sRGB
        terminology to linear (sRGB is the default, linear colorspace
        rendering may not be supported). Change extension name
        accordingly.
    Version 2, 2006/09/26 - add _OES extension suffix to bitfield names.
    Version 1, 2006/09/26 - first draft.
