# AMD_compressed_3DC_texture

Name

    AMD_compressed_3DC_texture

Name Strings

    GL_AMD_compressed_3DC_texture

Contributors

    Aaftab Munshi

Contact

    Benj Lipchak, AMD (benj.lipchak 'at' amd.com)

IP Status

    Please contact AMD regarding any intellectual property questions/issues 
    associated with this extension.

Status

    Complete.

Version

    Last Modified Date: February 26, 2008
    Revision: 6

Number

    OpenGL ES Extension #39

Dependencies

    Written based on the wording of the OpenGL ES 1.1 specification.

Overview

    Two compression formats are introduced:

    - A compression format for two component textures.  When used to store 
      normal vectors, the two components are commonly used with a fragment 
      shader that derives the third component.

    - A compression format for single component textures.  The single component
      may be used as a luminance or an alpha value.

    There are a large number of games that use luminance only and/or alpha only 
    textures.  For example, monochrome light maps used in a few popular games 
    are 8-bit luminance textures.  This extension describes a compression format
    that provides a 2:1 compression ratio for 8-bit single channel textures.

    Normal maps are special textures that are used to add detail to 3D surfaces.
    They are an extension of earlier "bump map" textures, which contained per-
    pixel height values and were used to create the appearance of bumpiness on 
    otherwise smooth surfaces.  Normal maps contain more detailed surface 
    information, allowing them to represent much more complex shapes.

    Normal mapping is one of the key features that makes the current generation
    of games look so much better than earlier titles.  A limitation to the 
    effectiveness of this technique is the size of the textures required.  In an 
    ideal case where every surface has both a color texture map and a normal 
    texture map, the texture memory and bandwidth requirements would double 
    compared to using color maps alone.

    In fact, the problem is much worse because existing block based compression
    methods such as DXTc, ETC, and S3TC are ineffective at compressing normal 
    maps.  They tend to have trouble capturing the small edges and subtle 
    curvature that normal maps are designed to capture, and they also introduce 
    unsightly block artifacts.

    Because normal maps are used to capture light reflections and realistic 
    surface highlights, these problems are amplified relative to their impact on
    color textures.  The results are sufficiently poor that game artists and 
    developers would rather not use normal map compression at all on most 
    surfaces, and instead limit themselves to lower resolution maps on selected 
    parts of the rendered scene.

    3DC provides an ideal solution to the normal map compression problem.  It 
    provides up to 4:1 compression of normal maps, with image quality that is 
    virtually indistinguishable from the uncompressed version.  The technique is
    hardware accelerated, so the performance impact is minimal.  Thus, 
    developers are freed to use higher resolution, more detailed normal maps, 
    and/or use them on all of the objects in a scene rather than just a select 
    few.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <internalFormat> parameter of CompressedTexImage2D and
    CompressedTexImage3DOES: 

        3DC_X_AMD             0x87F9
        3DC_XY_AMD            0x87FA

Additions to Chapter 3 of the OpenGL 1.3 Specification (Rasterization)

    Add to Table 3.17:  Specific Compressed Internal Formats

        Compressed Internal Format         Base Internal Format
        ==========================         ====================
        3DC_X_AMD                          RGB
        3DC_XY_AMD                         RGB


    Add to Section 3.8.3, Alternate Image Specification

    If <internalFormat> is 3DC_X_AMD, the compressed texture is a 
    single channel compressed texture.  If <internalFormat> is 3DC_XY_AMD,
    the compressed textures contains two channels.

    The details of these formats is not disclosed, so refer to AMD's 
    Compressonator tool in order to encode your textures offline:
    http://ati.amd.com/developer/compressonator.html
    
    3DC_X_AMD Format
    ================

    This format compresses a 128 bit block into 64 bits, representing a 2:1
    compression ratio.  The texture lookup unit will return (x, 0, 0, 1): the 
    decoded X value in the red component, 0.0 in the green and blue components,
    and 1.0 in the alpha component.

    3DC_XY_AMD Format
    =================

    This format compresses a 512 bit block into 128 bits, representing a 4:1
    compression ratio.  The texture lookup unit will return (x, y, 0, 1): the 
    decoded X value in the red component, the decoded Y value in the green
    component, 0.0 in the blue component, and 1.0 in the alpha component.

    Using 3DC_XY_AMD to compress normal maps requires an additional step. This 
    is because each value in a normal map is actually a 3D vector, consisting of
    3 components (x, y, z).  These values must be reduced to 2-component values
    in order to work with 3DC_XY_AMD.  Fortunately, this can be handled in a 
    simple way by assuming that all of the normal vectors have a length of 1.  
    Given the values of two components of a vector, the value of the third 
    component can be found using the following mathematical relationship: 
    z = sqrt(1 - (x*x + y*y)).  This formula can be implemented using just a
    couple of fragment shader instructions.

Errors

    INVALID_OPERATION is generated by TexImage2D, TexSubImage2D, 
    CompressedTexSubImage2D, or CopyTexSubImage2D if <internalformat> or 
    <format> is 3DC_X_AMD or 3DC_XY_AMD.

New State

    The queries for NUM_COMPRESSED_TEXTURE_FORMATS and 
    COMPRESSED_TEXTURE_FORMATS include 3DC_X_AMD and 3DC_XY_AMD.

Revision History

    02/26/2008    Benj Lipchak     Throw INVALID_OPERATION on subimage updates.
    09/24/2007    Jon Leech        Assign extension number.
    09/05/2007    Benj Lipchak     Cosmetic changes.
    08/01/2007    Benj Lipchak     Publication readiness.
    06/02/2006    Aaftab Munshi    Added IP status.
    05/12/2006    Aaftab Munshi    First Draft.
