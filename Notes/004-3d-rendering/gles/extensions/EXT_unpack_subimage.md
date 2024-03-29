# EXT_unpack_subimage

Name

    EXT_unpack_subimage

Name Strings

    GL_EXT_unpack_subimage

Contact

    Peter Pipkorn, NVIDIA Corporation (ppipkorn 'at' nvidia.com)

Contributors

    Daniel Kartch, NVIDIA Corporation (dkartch 'at' nvidia.com)
    Nicolai de Haan, NVIDIA Corporation (nicolaid 'at' nvidia.com)
    Daniel Koch, NVIDIA Corporation (dkoch 'at' nvidia.com)

Status

    Complete.

Version

    Last Modifed Date:  Feb 8, 2013
    Author Revision:    2

Number

    OpenGL ES Extension #90

Dependencies

    The extension is written against the OpenGL ES 2.0 specification.
    The extension references the OpenGL 2.0 specification.

Overview

    This OpenGL ES 2.0 extension adds support for GL_UNPACK_ROW_LENGTH,
    GL_UNPACK_SKIP_ROWS and GL_UNPACK_SKIP_PIXELS as valid enums to
    PixelStore.  The functionality is the same as in OpenGL. These are
    useful for updating textures with a sub-rectangle of pixel data
    coming from a larger image in host memory.

IP Status

    None

New Procedures and Functions

    None

New Tokens

    Accepted by the <pname> parameters of PixelStorei, GetIntegerv, and
    GetFloatv:

        GL_UNPACK_ROW_LENGTH_EXT            0x0CF2
        GL_UNPACK_SKIP_ROWS_EXT             0x0CF3
        GL_UNPACK_SKIP_PIXELS_EXT           0x0CF4

Additions to the OpenGL ES 2.0 Specification

    Modifications to Table 3.1 (PixelStore parameters)

        Add the following entries:

        Parameter Name          Type      Initial Value  Valid Range
        ==============          ====      =============  ===========
        UNPACK_ROW_LENGTH_EXT   integer   0              [0,Infinity)
        UNPACK_SKIP_ROWS_EXT    integer   0              [0,Infinity)
        UNPACK_SKIP_PIXELS_EXT  integer   0              [0,Infinity)

    Modifications to 3.6.2 Transfer of Pixel Rectangles, in the Unpacking
    section:

        Change

          "The number of groups in a row is width;"

        to

          "If the value of UNPACK_ROW_LENGTH_EXT is not positive, then the
          number of groups in a row is <width>; otherwise the number of
          groups is UNPACK_ROW_LENGTH_EXT."

        After the sentence

          "If the number of bits per element is not 1, 2, 4 or 8 times
          the number of bits in a GL ubyte, then k = nl for all values
          of a."

        insert:

          "There is a mechanism for selecting a sub-rectangle of groups
          from a larger containing rectangle. This mechanism relies on
          three integer parameters: UNPACK_ROW_LENGTH_EXT, UNPACK_SKIP_ROWS_EXT,
          and UNPACK_SKIP_PIXELS_EXT. Before obtaining the first group from
          memory, the pointer supplied to TexImage2D is effectively
          advanced by (UNPACK_SKIP_PIXELS_EXT)n + (UNPACK_SKIP_ROWS_EXT)k
          elements. Then <width> groups are obtained from contiguous
          elements in memory (without advancing the pointer), after
          which the pointer is advanced by k elements. <height> sets of
          <width> groups of values are obtained this way. See figure
          3.6."

        Before Table 3.5 Packed pixel formats, insert

           Figure 3.8 from the OpenGL 2.0 specification (a visual
           description of UNPACK_ROW_LENGTH_EXT, UNPACK_SKIP_ROWS_EXT, and
           UNPACK_SKIP_PIXELS_EXT)

Errors

    None

New State

    Modifications to Table 6.12 Pixels in section 6.2 State Tables:

        Get Value              Type  Get Cmnd     Initial  Description                     Sec.
                                                  Value
        ====================   ====  ===========  =======  =============================== =====
        UNPACK_ROW_LENGTH_EXT  Z+    GetIntegerv  0        Value of UNPACK_ROW_LENGTH_EXT  3.6.1
        UNPACK_SKIP_ROWS_EXT   Z+    GetIntegerv  0        Value of UNPACK_SKIP_ROWS_EXT   3.6.1
        UNPACK_SKIP_PIXELS_EXT Z+    GetIntegerv  0        Value of UNPACK_SKIP_PIXELS_EXT 3.6.1

Issues

    1. Can't this be done with repeated calls to
       TexSubImage2D/TexSubImage3D?

        Yes, it is possible to unpack pixels from a sub-rectangle in
        host memory by by calling these functions for one line at a
        time, but this could add unnecessary burden on the CPU system.
        Specifying GL_UNPACK_ROW_LENGTH_EXT makes it possible to unpack
        sub-rectangles of pixels with lower overhead.

    2. Should the corresponding PACK enums be added?

        No, it should be done in a separate extension. There is no
        dependency between the PACK enums and the UNPACK enums.

    3. Are these UNPACK_SKIP_* tokens strictly necessary?

        No. The same functionality can be achieved by advancing the
        pixel pointer to host memory appropriately before issuing an
        unpacking function call. They are included here for both
        completeness and for convenience.

    4. Should the new tokens be suffixed?
 
        Yes. This extension was originally drafted with unsuffixed tokens
        since they provide the same functionality as in core Desktop GL.
        However, the policy of the ES working group is that suffixes 
        must be used in extensions even for functionality that is core
        in Desktop GL.

Revision History

    Rev.   Date       Author       Changes
    ----   --------   ---------    ------------------------------------
     1     03/25/11   ppipkorn     First revision.
     2     02/08/13   dgkoch       add suffixes to tokens
                                   remove unnecessary column from table 6.12 edits
