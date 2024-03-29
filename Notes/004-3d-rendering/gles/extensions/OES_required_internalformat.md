# OES_required_internalformat

Name

    OES_required_internalformat

Name Strings

    GL_OES_required_internalformat

Contributors

    Aaftab Munshi
    Jeremy Sandmel
    Members of the Khronos OpenGL ES working group

Contact

    Benj Lipchak, Apple, Inc. (lipchak 'at' apple.com)
    
Notice

    Copyright (c) 2007-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Complete.
    Approved by the Khronos Promoters on April 26, 2012.

Version

    Last Modified Date: January 29, 2010
    Author Revision: 0.19

Number

    XXX TBD

Dependencies

    OpenGL ES 1.0 is required.

    This extension is written against the OpenGL ES 1.1 specification.

    OpenGL ES 2.0 affects the definition of this extension.
    
    OES_framebuffer_object affects the definition of this extension.

    OES_depth_texture affects the definition of this extension.
    
    OES_depth24 and OES_depth32 affect the definition of this extension.
    
    OES_packed_depth_stencil affects the definition of this extension.

    OES_rgb8_rgba8 affects the definition of this extension.

    OES_stencil1, OES_stencil4, and OES_stencil8 affect the definition of this
    extension.
    
    OES_texture_3D affects the definition of this extension.
    
    EXT_texture_type_2_10_10_10_REV affects the definition of this extension.

Overview

    The ES 1.1 API allows an implementation to store texture data internally
    with arbitrary precision, regardless of the format and type of the data
    supplied by the application.  Similarly, ES allows an implementation to
    choose an arbitrary precision for the internal storage of image data
    allocated by glRenderbufferStorageOES.

    While this allows flexibility for implementations, it does mean that an 
    application does not have a reliable means to request the implementation 
    maintain a specific precision or to find out what precision the 
    implementation will maintain for a given texture or renderbuffer image.

    For reference, "Desktop" OpenGL uses the <internalformat> argument to 
    glTexImage*, glCopyTexImage* and glRenderbufferStorageEXT as a hint, 
    defining the particular base format and precision that the application wants 
    the implementation to maintain when storing the image data.  Further, the 
    application can choose an <internalformat> with a different base internal 
    format than the source format specified by <format>.  The implementation is
    not required to exactly match the precision specified by <internalformat>
    when choosing an internal storage precision, but it is required to match the 
    base internal format of <internalformat>.

    In addition, ES 1.1 does not allow an implementation to fail a request to
    glTexImage2D for any of the legal <format> and <type> combinations listed in
    Table 3.4, even if the implementation does not natively support data stored
    in that external <format> and <type>.  However, there are no additional
    requirements placed on the implementation.  The ES implementation is free to
    store the texture data with lower precision than originally specified, for
    instance.  Further, since ES removes the ability to query the texture object
    to find out what internal format it chose, there is no way for the
    application to find out that this has happened.

    This extension addresses the situation in two ways:
    
        1) This extension introduces the ability for an application to specify
        the desired "sized" internal formats for texture image allocation.
    
        2) This extension guarantees to maintain at least the specified
        precision of all available sized internal formats.
        
    An implementation that exports this extension is committing to support all
    of the legal values for <internalformat> in Tables 3.4, 3.4.x, and 3.4.y,
    subject to the extension dependencies described herein.  That is to say, the 
    implementation is guaranteeing that choosing an <internalformat> argument
    with a value from these tables will not cause an image allocation request to
    fail.  Furthermore, it is guaranteeing that for any sized internal format,
    the renderbuffer or texture data will be stored with at least the precision
    prescribed by the sized internal format.
    
Issues

    1.  What API should we use to allow an application to indicate it wants 
        storage of at least a certain precision for a particular source format/
        type combination?
        
        RESOLVED
        
        We use the "sized" <internalformat> enums to indicate that an
        application desires some minimum precision for internal storage and 
        that the GL should not use less than the specified precision.

        We originally had considered using a new "UNPACK" parameter for
        glPixelStorei that indicated the external <format> and <type> arguments 
        must be supported with no "downsampling" of the precision.

        However, this latter approach ran into the problem that glCopyTexImage2D 
        and glRenderbufferStorageOES don't use the pixel store "UNPACK" state 
        nor do they have external <format> or <type> arguments to indicate the 
        application's requested precision.

        Another option was to create a new set of glTexImage*, glCopyTexImage*,
        and glRenderbufferStorageOES entry points that implied a minimum
        required precision.

        However, it seems the simplest thing to do is to just use the
        <internalformat> argument for this purpose and keep the existing entry 
        points.  It's also the most compatible with desktop OpenGL code.

    2.  Should this specification make any mention of "optionally supported"
        <internalformat> values at all?  Or should we just move all of those to 
        separate defined extensions?
        
        RESOLVED, all the values for <internalformat> in Tables 3.4, 3.4.x, and
        3.4.y in this extension must be accepted by their respective commands.
        All sized formats in these tables must be stored with at least the
        corresponding minimum precision.

        Other extensions may introduce new values for <internalformat>.  If they
        do, they should modify Tables 3.4, 3.4.x, and 3.4.y.  If an 
        implementation can support the new <internalformat> values, it will
        export the extension.
        
        The primary motivation for moving "optionally supported" formats to
        their own extensions is so that a well-behaved application never gets 
        the idea that it is supposed to check for errors in order to query for 
        available functionality.  Well-behaved applications should always query
        for extension strings and other implementation-dependent state to check
        for available functionality on a particular implementation.

    3.  What should be the recommended way to add new formats in the future?
    
        RESOLVED
        
        Just like before, new extensions can add new values for <internalformat> 
        for *TexImage* and RenderbufferStorageOES.  Unless otherwise stated by
        future extensions, new sized internal formats will provide the same kind
        of minimum precision guarantees as the formats described herein.

    4.  How should we handle render vs. texture buffers?
    
        RESOLVED
        
        We have three tables in the spec, one for textures specified with 
        TexImage*, one for textures specified with CopyTexImage2D, and one for 
        renderbuffers allocated with RenderbufferStorageOES.
        
    5.  Should this extension require RGBA8?  For both texture and render?
    
        RESOLVED
        
        Yes, for both.  OpenGL ES 2.0 implementations are very likely to have 
        this support anyway, and OpenGL ES 1.x implementations can choose to 
        export this extension or not, depending on whether they support 8 bit/
        component for textures and renderbuffers.
        
        Note, this extension does not require RGB8 for renderbuffers.  
        Availability of RGB8 renderbuffers is still based on presence of the
        OES_rgb8_rgba8 extension.
        
    6.  What should this extension say, if anything, about depth and stencil
        internal formats for RenderbufferStorageOES?
        
        RESOLVED
        
        Stencil and depth formats are listed in the Table 3.4.x.  If an
        implementation that supports this extension doesn't support one or more
        of the stencil/depth format extensions then this extension behaves as if 
        those enums are NOT listed in Table 3.4.x.
        
    7.  Should we allow every combination of <format>, <type>, and
        <internalformat>?
    
        RESOLVED
        
        No, we specifically disallow conversions between "base" formats (between 
        RGBA and LUMINANCE_ALPHA, for instance).  Further, we also disallow 
        requested "expansions" to larger data types.  That is, Table 3.4.x 
        allows the user to request that the GL use a lower precision than the 
        external <type> would require, but that table does not include any 
        entries that would require the GL to use more precision than the 
        external <type> would require.

        We intentionally don't include this feature here because the
        <internalformat> represents a required minimum precision for the GL.  We
        don't want to allow an application to require more precision internally
        than they provide externally because it would necessitate a format
        conversion.  If some implementation really wants to add this 
        functionality, it would need to create an additional extension, say 
        FOO_expanded_internalformat.

        Note that in the opposite situation where an application provides higher
        precision source data and asks it to be stored at lower precision, an
        implementation is still free to store at the higher precision, thereby
        avoiding format conversion.

    8.  Should we split this extension into multiple extensions?
    
        RESOLVED

        No.  No vendor has expressed interest in subsetting these 3 features:
        
            Feature 1:  This extension allows an application to specify sized 
            <internalformats> enums for texture and render buffer image 
            allocation.

            Feature 2:  This extension allows the implementation to export a 
            list of sized <internalformats> for which it will guarantee to 
            maintain the specified minimum precision when those formats are used 
            for texture or renderbuffer image allocation.

            Feature 3:  This extension defines a minimum set of sized
            <internalformat> enums that are required to be exported by the query 
            mechanism in #2 above.

    9.  Should we add a query that lets the implementation advertise which 
        sized internal formats it supports with minimum precision?

        RESOLVED 

        No, we'll leave it to a future extension to add this mechanism if
        necessary.  Today all sized internal formats have guaranteed minimum
        precision.  For posterity, we were close to choosing query mechanism 'c'
        from the list below before we dropped the query entirely:
         
            a) a query for a single format's support like this:
         
            boolean QuerySupportOES(enum param, int count, const enum* list); 

            which takes a <list> of <count> sized internal format enums and 
            returns GL_FALSE if any of them, when used as a sized internal 
            format with the currently bound context, would result in an internal 
            format of lower precision than requested.  The <param> could be
            GL_SUPPORTED_TEXTURE_INTERNALFORMAT for querying supported texture 
            precisions and GL_SUPPORTED_RENDERBUFFER_INTERNALFORMAT for querying
            supported renderbuffer precisions.
        
            b) a query for all supported formats that the implementation can 
               guarantee minimum precision, like this:
        
            void GetSupportedInternalFormatsOES(enum param, 
                                                int* count, 
                                                enum* list);
        
            where <param> must be TEXTURE_INTERNALFORMAT or
            RENDERBUFFER_INTERNALFORMAT.  On input <count> is the maximum
            number of internal formats to return, and on output <count>
            is the number of internal formats in <list>.  On output, <list> 
            contains all the values of <internalformat> for which the 
            implementation will guarantee specified precision.  <list> will 
            contain <count> entries.
            
            c) a query of a single internalformat providing a yes/no answer:

            boolean IsPreciseInternalformatOES(enum param, enum internalformat);

            where <param> must be TEXTURE or RENDERBUFFER_OES, and
            <internalformat> is the internal format for which the app is
            checking the minimum precision storage guarantee.

    10. What is the OES_framebuffer_object interaction?  Are FBOs a 
        prerequisite (even on ES 1.1), or should we strike mention of
        renderbuffers if FBOs aren't available?

        RESOLVED

        Sized format hints and minimum precision guarantees for textures are
        useful even on implementations where FBOs aren't supported.  We won't
        make FBOs a prerequisite. 

    11. Should we add retroactive support for EXT_texture_type_2_10_10_10_REV?

        RESOLVED

        Yes.  We introduced two new sized internal formats, RGB10_A2_EXT and
        RGB10_EXT.  This format continues to be unrenderable, consistent with
        the EXT_texture_type_2_10_10_10_REV spec.  These formats join the rest
        of existing sized internal formats on the required list of formats for
        which precision is guaranteed.

    12. Do we need different token values from OpenGL?

        RESOLVED

        No. Initially some new tokens (such as ALPHA8_OES) were given new
        values, which was due to a difference in the meaning relative to
        OpenGL: in OpenGL 2.1, sized internal formats were purely hints,
        whereas in this extension they are lower bounds. However, OpenGL
        now specifies a number of formats as being "required" in the
        same sense of a lower bound on precision, and the token values
        are not changed.

New Procedures and Functions

    None.

New Types

    None.

New Tokens
    
     Accepted by the <internalformat> argument of TexImage2D, TexImage3DOES, and 
     CopyTexImage2D:
         
         ALPHA8_OES                       0x803C
         LUMINANCE8_OES                   0x8040
         LUMINANCE8_ALPHA8_OES            0x8045
         LUMINANCE4_ALPHA4_OES            0x8043
         RGB565_OES                       0x8D62
         RGB8_OES                         0x8051
         RGBA4_OES                        0x8056
         RGB5_A1_OES                      0x8057
         RGBA8_OES                        0x8058
         DEPTH_COMPONENT16_OES            0x81A5
         DEPTH_COMPONENT24_OES            0x81A6
         DEPTH_COMPONENT32_OES            0x81A7
         DEPTH24_STENCIL8_OES             0x88F0
         RGB10_EXT                        0x8052
         RGB10_A2_EXT                     0x8059
         
     Accepted by the <internalformat> argument of RenderbufferStorageOES:
         
         RGBA8_OES                        0x8058
         
Additions to Chapter 2 of the OpenGL ES 1.1 Specification (OpenGL ES Operation)

    None.

Additions to Chapter 3 of the OpenGL ES 1.1 Specification (Rasterization)

    In section 3.6.2 ("Unpacking"), p.69 (p. 60/61 of ES 2.0 spec), modify
    Table 3.4 and replace the preceding paragraph as follows:


            Format            Type                            External Bytes per Pixel Internal format
            ------            ----                            ------------------------ ---------------
            RGBA              UNSIGNED_INT_2_10_10_10_REV_EXT 4                        RGBA, RGB10_A2_EXT, RGB5_A1
            RGBA              UNSIGNED_BYTE                   4                        RGBA, RGBA8, RGB5_A1, RGBA4  
            RGB               UNSIGNED_INT_2_10_10_10_REV_EXT 4                        RGB,  RGB10_EXT, RGB8, RGB565 
            RGB               UNSIGNED_BYTE                   3                        RGB,  RGB8, RGB565         
            RGBA              UNSIGNED_SHORT_4_4_4_4          2                        RGBA, RGBA4
            RGBA              UNSIGNED_SHORT_5_5_5_1          2                        RGBA, RGB5_A1
            RGB               UNSIGNED_SHORT_5_6_5            2                        RGB,  RGB565
            LUMINANCE_ALPHA   UNSIGNED_BYTE                   2                        LUMINANCE_ALPHA, LUMINANCE8_ALPHA8, LUMINANCE4_ALPHA4
            LUMINANCE         UNSIGNED_BYTE                   1                        LUMINANCE,  LUMINANCE8
            ALPHA             UNSIGNED_BYTE                   1                        ALPHA, ALPHA8
            DEPTH_COMPONENT   UNSIGNED_SHORT                  2                        DEPTH_COMPONENT, DEPTH_COMPONENT16_OES 
            DEPTH_COMPONENT   UNSIGNED_INT                    4                        DEPTH_COMPONENT, DEPTH_COMPONENT32_OES, DEPTH_COMPONENT24_OES, DEPTH_COMPONENT16_OES
            DEPTH_STENCIL_OES UNSIGNED_INT_24_8_OES           4                        DEPTH_STENCIL_OES, DEPTH24_STENCIL8_OES

            Table 3.4: Valid combinations of <format>, <type>, and
            <internalformat> for TexImage2D and TexImage3DOES

        When calling TexImage2D or TexImage3DOES, not all combinations of 
        <format>, <type>, and <internalformat> are valid.  The valid 
        combinations accepted by the GL are defined in Table 3.4.  If TexImage2D 
        or TexImage3DOES is called with a combination of <format>, <type>, and 
        <internalformat> not listed in Table 3.4, then INVALID_OPERATION is 
        generated.
        
        In addition, only certain values for <internalformat> are valid when 
        calling RenderbufferStorageOES and CopyTexImage2D.  The valid values of 
        <internalformat> are listed in Tables 3.4.x and 3.4.y.  If
        RenderbufferStorageOES is called with a value of <internalformat> not
        listed in Table 3.4.x then INVALID_ENUM is generated.  Similarly, if
        CopyTexImage2D is called with a value of <internalformat> not listed in
        Table 3.4.y, then INVALID_ENUM is generated.

            Internal format
            ---------------
            RGBA4_OES
            RGB5_A1_OES
            RGBA8_OES
            RGB565_OES
            RGB8_OES
            STENCIL_INDEX1_OES
            STENCIL_INDEX4_OES
            STENCIL_INDEX8_OES
            DEPTH_COMPONENT16_OES
            DEPTH_COMPONENT24_OES
            DEPTH_COMPONENT32_OES
            DEPTH24_STENCIL8_OES

            Table 3.4.x:  Legal values of <internalformat> for
            RenderbufferStorageOES

            Internal format
            ---------------    
            RGBA
            RGBA4_OES
            RGB5_A1_OES
            RGBA8_OES
            RGB10_A2_EXT
            RGB
            RGB565_OES
            RGB8_OES
            RGB10_EXT
            LUMINANCE_ALPHA
            LUMINANCE4_ALPHA4_OES
            LUMINANCE8_ALPHA8_OES
            LUMINANCE
            LUMINANCE8_OES
            ALPHA
            ALPHA8_OES

            Table 3.4.y:  Legal values of <internalformat> for CopyTexImage2D

        An implementation must accept all of the values for <internalformat> 
        specified in Tables 3.4, 3.4.x, 3.4.y.  Furthermore, an implementation
        must respect the minimum precision requirements of sized internal
        formats -- those with explicit component resolutions -- by storing each
        component with at least the number of bits prescribed.

        If one of the base "unsized" formats (RGBA, RGB, LUMINANCE_ALPHA, 
        LUMINANCE, ALPHA, DEPTH_COMPONENT, or DEPTH_STENCIL_OES) is specified
        for <internalformat> to TexImage2D, TexImage3DOES, or CopyTexImage2D,
        the GL is free to choose the precision that it will maintain for the
        texture.  ES implementations are still encouraged, however, to maintain
        as much precision as possible, given the source arguments to those
        commands.

    In section 3.7.9 (3.7.10 for ES 2.0), update the definition of texture
    completeness (and cube completeness for ES 2.0 or in the presence of
    OES_texture_cube_map) to read:

        "... were each specified with the same format, type, and internal
        format."
        
Additions to Chapter 4 of the OpenGL ES 1.1 Specification (Per-Fragment 
Operations and the Frame Buffer)

    None.

Additions to Chapter 5 of the OpenGL ES 1.1 Specification (Special Functions)

    None.

Additions to Chapter 6 of the OpenGL ES 1.1 Specification (State and State 
Requests)

    None.

Additions to Appendix A of the OpenGL ES 1.1 Specification (Invariance)

    None.

Additions to the AGL/EGL/GLX/WGL Specifications

    None.

GLX Protocol

    XXX TBD

Errors

   If TexImage2D or TexImage3DOES is called with a combination of <format>, 
   <type>, and <internalformat> not listed in Table 3.4, then INVALID_OPERATION
   is generated.

   If RenderbufferStorageOES or CopyTexImage2D is called with a value of
   <internalformat> not listed in Table 3.4.x or 3.4.y, respectively, then 
   INVALID_ENUM is generated.
   
New State

    None.

New Implementation Dependent State

    None.
    
Dependencies on OpenGL ES 2.0

    If OpenGL ES 2.0 is supported, the following tokens do not have the
    "_OES" suffix: RGB565, RGBA4, RGB5_A1, DEPTH_COMPONENT16, STENCIL_INDEX8,
    and RENDERBUFFER.  Also, RenderbufferStorage does not have the "OES" suffix.

Dependencies on OES_depth_texture
    
    If OES_depth_texture is not supported, then all references to
    DEPTH_COMPONENT as a legal value for <format> when calling TexImage2D and 
    TexImage3DOES should be deleted.

Dependencies on OES_depth24 and OES_depth32

    If OES_depth24 is not supported, then all references to
    DEPTH_COMPONENT24_OES should be deleted.

    If OES_depth32 is not supported, then all references to
    DEPTH_COMPONENT32_OES should be deleted.

Dependencies on OES_framebuffer_object

    If OES_framebuffer_object is not supported and OpenGL ES 2.0 is not 
    supported, then all references to RENDERBUFFER_OES and renderbuffers should
    be deleted, including Table 3.4.x.

Dependencies on OES_packed_depth_stencil
    
    If OES_packed_depth_stencil is not supported, then all references to
    DEPTH_STENCIL_OES, DEPTH24_STENCIL8_OES, and UNSIGNED_INT_24_8_OES should be 
    deleted.

Dependencies on OES_rgb8_rgba8

    If OES_rgb8_rgba8 is not supported, then references to RGB8_OES as a 
    renderbuffer format in Table 3.4.x should be deleted.

Dependencies on OES_stencil1, OES_stencil4, and OES_stencil8 

    If OES_stencil1 is not supported, then all references to STENCIL_INDEX1_OES
    should be deleted.

    If OES_stencil4 is not supported, then all references to STENCIL_INDEX4_OES
    should be deleted.

    If OES_stencil8 is not supported and OpenGL ES 2.0 is not supported, then 
    all references to STENCIL_INDEX8_OES should be deleted.

Dependencies on OES_texture_3D

    If OES_texture_3D is not supported, then all references to TexImage3DOES 
    should be deleted.

Dependencies on EXT_texture_type_2_10_10_10_REV

    If EXT_texture_type_2_10_10_10_REV is not supported, then all references to
    RGB10_A2_EXT, RGB10_EXT, and UNSIGNED_INT_2_10_10_10_REV_EXT should be
    deleted.

Sample Code

    // ====================================================================
    // Example 1: 
    // An application that requires support for RGBA with >= 8888 storage
    // can indicate so by simply checking for the OES_required_internalformat
    // extension string, since RGBA8_OES availability is a prerequisite.
    // ====================================================================
    
    // First, check for presence of this extension.
    bool reqSupportAvailable = 
        MyExtensionQuery("GL_OES_required_internalformat");
    if (!reqSupportAvailable)
    {
        printf("No support for GL_OES_required_internalformat!\n");
        exit(1);
    }

    // This texture is guaranteed to be stored with at least 8-bit per component
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8_OES, 32, 32, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, myTexelData);

    // ====================================================================
    // Example 2: 
    // An application that requires support for DEPTH_COMPONENT with >= 32 bit
    // storage can indicate so by checking for both OES_required_internalformat
    // and OES_depth32 extension strings.  That will guarantee minimum precision
    // for 32-bit depth values stored in renderbuffers.  If the application also
    // needs 32-bit depth stored in textures, it can check for the 
    // OES_depth_texture extension, too.
    // ====================================================================

    // First, check for presence of this extension.
    bool reqSupportAvailable = CheckExtension("GL_OES_required_internalformat");
    if (!reqSupportAvailable)
    {
        printf("No support for GL_OES_required_internalformat.\n");
        printf("Minimum precision guarantees cannot be made!\n");
        exit(1);
    }
    
    // Now check for the presence of the depth32 extension.
    bool depth32_available = CheckExtension("GL_OES_depth32");
    if (!depth32_available)
    {
        printf("No support for OES_depth32.\n");
        printf("32-bit depth renderbuffers not available!\n");
        exit(1);
    }

    // We're now guaranteed minimum 32-bit depth renderbuffers
    glRenderbufferStorageOES(GL_RENDERBUFFER_OES, GL_DEPTH_COMPONENT32_OES,
                             32, 32);

    // Now check for the presence of the depth texture extension.
    bool depth_texture_available = CheckExtension("GL_OES_depth_texture");
    if (!depth_texture_available)
    {
        printf("No support for OES_depth_texture.\n");
        printf("32-bit depth textures not available!\n");
        exit(1);
    }

    // We're now guaranteed minimum 32-bit depth textures
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32_OES, 32, 32, 0,
                 GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, myD32TexelData);

Revision History

    0.19, 01/29/2010, bmerry
    - changed token values for six formats where they were different
      from equivalent OpenGL tokens, and added issue 12

    0.18, 11/26/2008, benj
    - RGBA8 is mandated as a renderbuffer format, while RGB8 is only available
      if OES_rgb8_rgba8 is present.

    0.17, 11/19/2008, benj
    - treat OES_rgb8_rgba8 as optional like other renderbuffer format extensions
    - update Dependencies section
    - assign enum values for new tokens
    - fix minor typos

    0.16, 11/03/2008, benj
    - remove query mechanism since all sized internal formats have guaranteed
      minimum precision today
    - mention OES_rgb8_rgba8 requirement under Dependencies section
    - remove more tokens introduced by other extensions from New Tokens
    - update overview, issues list, and sample code

    0.15, 10/29/2008, benj
    - fix sample code bugs reported by Acorn
    - firm up the definition of texture completeness, requested by Acorn

    0.14, 10/29/2008, benj
    - fix spec bug reported by Ben in New Tokens section: split up token list
    - remove unrenderable RGB10_A2_EXT and RGB10_EXT formats from Table 3.4.x
    - get rid of gl, GL, and GL_ prefixes
    - switch query mechanism to new glIsRespectedInternalFormat
    - update issues list and sample code

    0.13, 10/28/2008, benj
    - major overhaul
    - remove double pointer from query command
    - fix extension names for depth and stencil extensions
    - sync up with latest 10/10/10/2 texture and depth texture extensions
    - don't allow RGBA8 or RGBA4 internal formats for 10/10/10/2 textures
      since that would require expanding the alpha channel
    - add OES suffix to most tokens, EXT for 10/10/10/2 tokens
    - remove tokens that aren't introduced by this extension from New Tokens
    - add missing ALPHA8_OES and DEPTH_COMPONENT16_OES to New Tokens
    - remove unsized formats from Table 3.4.x, add missing DEPTH_COMPONENT16
    - remove L, A, and LA formats from Table 3.4.x and Table 3.6.x
      RENDERBUFFER_OES section
    - add interactions with ES 2.0 and OES_framebuffer_object
    - change error for invalid combination of format/type/internalformat from
      INVALID_ENUM to INVALID_OPERATION
    - reformat for 80 columns
    - update issues list and sample code

    0.12, 02/27/2008, jsandmel
    - added glGetSupportedInternalFormatsOES to new procedures section
    - fixed typo in argument name for <internalformatlist>

    0.11, 02/05/2008, jsandmel
    - reworked API after the Portland F2F
    - added query for supported internal formats
    - updated section 3.6.2.1 on how the required list works
    - added table 3.6.x for the list of required <internalformats>
    - added a sample code example of the query for supported <internalformats>
    - added dependency on packed depth stencil

    0.10, 10/16/2007, jsandmel
    - fixed several typos and non-ascii characters, no functional changes

    0.9, 10/15/2007, jsandmel
    - added issues 12 and 13
    - included Acorn's suggestion for a query for supported formats that
      respect the minimum precision guarantee

    0.8, 05/27/2007, jsandmel
    - added contributors section
    - updated spec with recent group decisions
    - added explicit dependencies on OES_stencil, OES_depth extensions
    - split table 3.4.x into two tables: one for RenderbufferStorageOES
      and one for CopyTexImage since you can create a depth/stencil
      renderbuffer but not texture.

    0.7, 05/21/2007, jsandmel
    - updated spec with recent group decisions
    - removed concept of "optionally supported" extensions
    - added second table for RenderBuffer storage

    0.6, 05/03/2007, jsandmel
    - fixed some typos refering to OpenGL 1.5 spec that should be ES 1.1 spec

    0.5, 4/28/2007, jsandmel
    - changed a few more references to LUMINANCE/ALPHA instead of L/A
    - also clarified overview wrt to required <internalformat> semantics

    0.4, 4/25/2007, jsandmel
    - corrected table 3.4 and 3.4x to use LUMINANCE/ALPHA instead of L/A

    0.3, 4/24/2007, jsandmel
    - added issue 5 to discuss whether we should move all the "optionally
      supported" <internalformat> values to their own extensions

    0.2, 4/24/2007, jsandmel
    - fixed up a few typos and lingering references to UNPACK state

    0.1, 4/23/2007, jsandmel
    - switch to start using <internalformat> instead of adding a new
      UNPACK parameter.
    
    0.0, 4/17/2007, jsandmel
    - initial revision
