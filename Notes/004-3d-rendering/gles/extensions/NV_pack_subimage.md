# NV_pack_subimage

Name

    NV_pack_subimage

Name Strings

    GL_NV_pack_subimage

Contact

    Peter Pipkorn, NVIDIA Corporation (ppipkorn 'at' nvidia.com)

Contributors

    Pat Brown, NVIDIA
    Nicolai de Haan, NVIDIA
    Greg Roth, NVIDIA

Status

    Complete.

Version

    Last Modifed Date:  Oct 07, 2012
    Author Revision:    3

Number

    OpenGL ES Extension #132

Dependencies

    OpenGLES 2.0 is required.    

    The extension is written against the OpenGL ES 2.0.25 specification.

    EXT_unpack_subimage is required.
    
Overview

    This OpenGL ES 2.0 extension adds support for GL_PACK_ROW_LENGTH_NV,
    GL_PACK_SKIP_ROWS_NV and GL_PACK_SKIP_PIXELS_NV as valid enums to
    PixelStore. The functionality is the same as in OpenGL. These are
    useful to update a sub-rectangle in host memory with data that can
    be read from the framebuffer or a texture (using FBO and texture
    attachments).

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameters of PixelStorei,
    GetIntegerv, and GetFloatv:

        GL_PACK_ROW_LENGTH_NV             0x0D02
        GL_PACK_SKIP_ROWS_NV              0x0D03
        GL_PACK_SKIP_PIXELS_NV            0x0D04

Additions to the OpenGL ES 2.0 Specification

    Modifications to Table 3.4 (PixelStore parameters for ReadPixels)

    Add the following entries:

        Parameter Name          Type      Initial Value  Valid Range
        ==============          ====      =============  ===========
        PACK_ROW_LENGTH_NV      integer   0              [0,Infinity)
        PACK_SKIP_ROWS_NV       integer   0              [0,Infinity)
        PACK_SKIP_PIXELS_NV     integer   0              [0,Infinity)


    Note: The description of the behavior of the added PACK* PixelStore
    parameters is covered by the following existing text from 4.3.1
    subsection "Placement in Client Memory":

        Groups of elements are placed in memory just as they are taken
        from memory for TexImage2D. That is, the ith group of the jth
        row (corresponding to the ith pixel in the jth row) is placed in
        memory just where the ith group of the jth row would be taken
        from for TexImage2D. See Unpacking under section 3.6.2. The only
        difference is that the storage mode parameters whose names begin
        with PACK_ are used instead of those whose names begin with
        UNPACK_.

    The equivalent UNPACK_ storage mode parametes are documented by the
    text added by EXT_unpack_subimage. As such, no additional
    documentation language is required here.

Errors

    None

New State

    Modifications to Table 6.12 Pixels in section 6.2 State Tables:

        Get Value              Type  Get Cmnd     Initial  Description                   Sec.    Attribute
                                                  Value
        ====================   ====  ===========  =======  ============================  =====   ===========
        PACK_ROW_LENGTH_NV     Z+    GetIntegerv  0        Value of PACK_ROW_LENGTH_NV   4.3.1   pixel-store
        PACK_SKIP_ROWS_NV      Z+    GetIntegerv  0        Value of PACK_SKIP_ROWS_NV    4.3.1   pixel-store
        PACK_SKIP_PIXELS_NV    Z+    GetIntegerv  0        Value of PACK_SKIP_PIXELS_NV  4.3.1   pixel-store


Issues

    1. Can't this be done with repeated calls to ReadPixels?

        RESOLVED: Yes, it is possible to pack pixels into a sub-
        rectangle in host memory by by calling this function for one
        line at a time with <height> of 1 advancing the <data> pointer
        each time, but this could add unnecessary burden on the CPU
        system. Specifying GL_PACK_ROW_LENGTH_NV makes it possible to
        pack sub-rectangles of pixels with lower overhead.

    2. Should the corresponding UNPACK enums be added?

        RESOLVED: No, it should be done in a separate extension. There
        is no functional dependency between the PACK enums and the
        UNPACK enums. However, there is a language dependency. This
        extension extends the language added by EXT_unpack_subimage.
        Since this is intended to ship on platforms that support both,
        Nothing is lost by adding a dependency for this reason.

    3. Are these PACK_SKIP_* tokens strictly necessary?

        RESOLVED: No. The same functionality can be achieved by
        advancing the pixel pointer to host memory appropriately before
        issuing an packing function call. They are included here for
        both completeness and for convenience.

    4. Should PACK_SKIP_IMAGES and PACK_IMAGE_HEIGHT be included?

        RESOLVED: No. Without support for GetTexImage, their inclusion
        makes less sense. The UNPACK_* equivalents were also left out
        of EXT_unpack_subimage, which makes adding them here more
        complicated to do right.

Revision History
    Rev.    Date          Author       Changes
    ----   ------------   ---------    -------------------------------------
     3     07 Nov 2012    groth        Added issue clarifications and a note
                                       about existing spec language.
     2     23 Oct 2012    groth        Removed references to 3D texture images.
     1     02 Oct 2009    ppipkorn     Original draft.

