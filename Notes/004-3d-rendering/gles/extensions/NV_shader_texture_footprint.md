# NV_shader_texture_footprint

Name

    NV_shader_texture_footprint

Name Strings

    GL_NV_shader_texture_footprint

Contact

    Chris Lentini, NVIDIA (clentini 'at' nvidia.com)
    Pat Brown, NVIDIA (pbrown 'at' nvidia.com)

Contributors

    Jeff Bolz, NVIDIA
    Daniel Koch, NVIDIA

Status

    Shipping

Version

    Last Modified Date:         February 8, 2018
    NVIDIA Revision:            2

Number

    OpenGL Extension #530
    OpenGL ES Extension #313

Dependencies

    This extension is written against the OpenGL 4.6 Specification
    (Compatibility Profile), dated July 30, 2017.

    OpenGL 4.5 or OpenGL ES 3.2 is required.

    This extension requires support for the OpenGL Shading Language (GLSL)
    extension "NV_shader_texture_footprint", which can be found at the Khronos
    Group Github site here:

        https://github.com/KhronosGroup/GLSL

Overview

    This extension adds OpenGL API support for the OpenGL Shading Language
    (GLSL) extension "NV_shader_texture_footprint".  That extension adds a new
    set of texture query functions ("textureFootprint*NV") to GLSL.  These
    built-in functions prepare to perform a filtered texture lookup based on
    coordinates and other parameters passed in by the calling code.  However,
    instead of returning data from the provided texture image, these query
    functions instead return data identifying the _texture footprint_ for an
    equivalent texture access.  The texture footprint identifies a set of
    texels that may be accessed in order to return a filtered result for the
    texture access.

    The footprint itself is a structure that includes integer values that
    identify a small neighborhood of texels in the texture being accessed and
    a bitfield that indicates which texels in that neighborhood would be used.
    Each bit in the returned bitfield identifies whether any texel in a small
    aligned block of texels would be fetched by the texture lookup.  The size
    of each block is specified by an access _granularity_ provided by the
    shader.  The minimum granularity supported by this extension is 2x2 (for
    2D textures) and 2x2x2 (for 3D textures); the maximum granularity is
    256x256 (for 2D textures) or 64x32x32 (for 3D textures).  Each footprint
    query returns the footprint from a single texture level.  When using
    minification filters that combine accesses from multiple mipmap levels,
    shaders must perform separate queries for the two levels accessed ("fine"
    and "coarse").  The footprint query also returns a flag indicating if the
    texture lookup would access texels from only one mipmap level or from two
    neighboring levels.

    This extension should be useful for multi-pass rendering operations that
    do an initial expensive rendering pass to produce a first image that is
    then used as a texture for a second pass.  If the second pass ends up
    accessing only portions of the first image (e.g., due to visibility), the
    work spent rendering the non-accessed portion of the first image was
    wasted.  With this feature, an application can limit this waste using an
    initial pass over the geometry in the second image that performs a
    footprint query for each visible pixel to determine the set of pixels that
    it needs from the first image.  This pass would accumulate an aggregate
    footprint of all visible pixels into a separate "footprint texture" using
    shader atomics.  Then, when rendering the first image, the application can
    kill all shading work for pixels not in this aggregate footprint.

    The implementation of this extension has a number of limitations.  The
    texture footprint query functions are only supported for two- and
    three-dimensional textures (TEXTURE_2D, TEXTURE_3D).  Texture footprint
    evaluation only supports the CLAMP_TO_EDGE wrap mode; results are
    undefined for all other wrap modes.  The implementation supports only a
    limited set of granularity values and does not support separate coverage
    information for each texel in the original texture.


New Procedures and Functions

    None

New Tokens

    None

Additions to Chapter 8 of the OpenGL 4.6 (Compatibility Profile) Specification
(Textures and Samplers)

    (add a new section immediately after section 8.15, Texture Magnification)

    Section 8.X, Texture Footprint Queries

    The OpenGL Shading Language provides a collection of built-in functions,
    all beginning with "textureFootprint", that allow shaders to query a
    _texture footprint_.  The texture footprint is a set of texels belonging
    to a single texture level that would be accessed when performing a
    filtered texture lookup.  The shader code calling the footprint query
    functions passes in a _granularity_ value, which is used to subdivide a
    texture level into an array of fixed-size _texel groups_ whose size is
    given by the granularity.  The texture footprint query functions return
    the footprint using a built-in GLSL data structure that identifies the set
    of texel groups containing one or more texels that would be accessed in an
    equivalent texture lookup.  Texture footprint queries are only supported
    for two- and three-dimensional textures (targets TEXTURE_2D and
    TEXTURE_3D).  Additionally, footprint queries require the use of the
    CLAMP_TO_EDGE sampler wrap mode in all relevant dimensions; the results of
    the footprint query are undefined if any other wrap mode is used.

    Each texture footprint query built-in function accepts a set of texture
    coordinates and any additional parameters (e.g., explicit level of detail,
    level of detail bias, or derivatives) needed to specify a normal texture
    lookup operation whose footprint should be evaluated.  The footprint query
    functions also accept a <granularity> parameter and a <coarse> flag used
    to select the level of detail whose footprint is returned.  The
    granularity parameter identifies the size of the texel groups used for the
    footprint query as described in Table X.1.  The <coarse> flag is used to
    select between the two levels of detail used when minifying using a filter
    (NEAREST_MIPMAP_LINEAR, LINEAR_MIPMAP_LINEAR) that averages texels from
    multiple levels of detail.  When such minification is performed, a value
    of "false" requests the footprint in the higher-resolution (fine) level of
    detail, while "true" requests the footprint in the lower-resolution
    (coarse) level of detail.  When a texture access uses only a single level
    of detail, its footprint will be returned for queries with <coarse> set to
    false, while queries with <coarse> set to true will return an empty
    footprint.  Since many texture accesses may use only a single level, the
    footprint query functions return a boolean value, which will be true if
    and only if all accessed texels are in a single level of detail.

      Granularity Value |  TEXTURE_2D   |  TEXTURE_3D
      ------------------+---------------+----------------
              0         |  unsupported  |  unsupported
              1         |      2x2      |     2x2x2
              2         |      4x2      |  unsupported
              3         |      4x4      |     4x4x2
              4         |      8x4      |  unsupported
              5         |      8x8      |  unsupported
              6         |     16x8      |  unsupported
              7         |     16x16     |  unsupported
              8         |  unsupported  |  unsupported
              9         |  unsupported  |  unsupported
              10        |  unsupported  |    16x16x16
              11        |     64x64     |    32x16x16
              12        |    128x64     |    32x32x16
              13        |    128x128    |    32x32x32
              14        |    256x128    |    64x32x32
              15        |    256x256    |  unsupported

      Table X.1:  Supported granularities for texture footprint queries, for
      two-dimensional (TEXTURE_2D) and three-dimensional (TEXTURE_3D)
      accesses.  Granularity values not listed in the table or listed as
      "unsupported" are not supported by this extension and result in
      undefined behavior if used.

      In addition to the boolean result, texture footprint queries return
      footprint data in a structure of the type gl_TextureFootprint2DNV (for
      two-dimensional textures) or gl_TextureFootprint3DNV (for
      three-dimensional textures).  In either structure, the member <lod>
      specifies the level-of-detail number used for the footprint.  The
      members <anchor> and <offset> identify a small neighborhood of texel
      groups in the texture used by the query.  The member <mask> specifies 64
      bits of data indicating which texel groups in the neighborhood are part
      of the footprint.  The member <granularity> returns information on the
      size of the texel groups in the footprint, which is sometimes larger
      than the requested granularity, as described below.

      For two-dimensional footprint queries, the neighborhood returned by the
      query is an 8x8 array of texel groups, where each texel group in
      neighborhood is identified by a coordinate (x,y), where <x> and <y> are
      integer values in the range [0,7].  Each texel group corresponds to a
      set of texels whose (u,v) coordinates satisfy the inequalities:

        u1 <= u <= u2
        v1 <= v <= v2

      computed using the following logic:

        // The footprint logic returns a mask whose bits are aligned to 8x8
        // sets of texel groups.  This allows shaders to use atomics to
        // efficiently accumulate footprint results across many invocations,
        // storing an 8x8 array of bits for each group into one RG32UI texel.
        // The texel group number in the neighborhood is treated as an offset
        // relative to the anchor point.
        uvec2 texel_group = 8 * result.anchor + uvec2(x,y);

        // If the neighborhood crosses the boundaries of an 8x8 set, the bits
        // of the mask are effectively split across multiple sets (up to
        // four for 2D).  The "offset" parameter returned by the query
        // identifies which x/y group values in the neighborhood are
        // assigned to which set.  An all-zero offset indicates that the
        // footprint is fully contained in a single 8x8 set at the anchor.
        // "Low" x/y values identify texel groups at the beginning of the 8x8
        // set identified by the anchor, while "high" values correspond to
        // texel groups at the end of the previous set.  The offset
        // indicates the number of texel groups assigned to the previous set.
        if (x + result.offset.x >= 8) {
            texel_group.x -= 8;
        }
        if (y + result.offset.y >= 8) {
            texel_group.y -= 8;
        }

        // Once we have a group number, u/v texel number ranges are generated by
        // multiplying by the texel group size.
        uint u1 = texel_group.x * granularity_x;
        uint u2 = u1 + granularity_x - 1;
        uint v1 = texel_group.y * granularity_y;
        uint v2 = v1 + granularity_y - 1;

      In the equations above, <granularity_x> and <granularity_y> refer to the
      texel group size as in Table X.1.  result.anchor and result.offset
      specify the <anchor> and <offset> members of the returned structure, and
      <x> and <y> specify the texel group number in the neighborhood.

      Each bit in the <mask> member of the returned structure corresponds to
      one of the texel groups in the 8x8 neighborhood.  That bit will be set
      if and only if any of the texels in the texel group is covered by the
      footprint.  The texel group (x,y) is considered to be covered if and
      only if the following logic computes true for <covered>:

        uint64_t mask = result.mask.x | (result.mask.y << 32);
        uint32_t bit = y * 8 + x;
        bool covered = (0 != ((mask >> bit) & 1));

      For three-dimensional footprint queries, the logic is very similar,
      except that the neighborhood returned by the query is a 4x4x4 array of
      texel groups.  Each texel group in neighborhood is identified by a
      coordinate (x,y,z), where <x>, <y>, and <z> are integer values in the
      range [0,3].  Each texel group corresponds to a set of texels whose
      (u,v,w) coordinates satisfy the inequalities:

        u1 <= u <= u2
        v1 <= v <= v2
        w1 <= w <= w2

      computed using the following logic:

        uvec3 texel_group = 4 * result.anchor + uvec3(x,y,z);
        if (x + result.offset.x >= 4) {
            texel_group.x -= 4;
        }
        if (y + result.offset.y >= 4) {
            texel_group.y -= 4;
        }
        if (z + result.offset.z >= 4) {
            texel_group.z -= 4;
        }
        uint u1 = texel_group.x * granularity_x;
        uint u2 = u1 + granularity_x - 1;
        uint v1 = texel_group.y * granularity_y;
        uint v2 = v1 + granularity_y - 1;
        uint w1 = texel_group.z * granularity_z;
        uint w2 = w1 + granularity_z - 1;

      As in the two-dimensional logic, <granularity_x>, <granularity_y>, and
      <granularity_z> refer to the texel group size as in Table X.1.
      result.anchor and result.offset specify the <anchor> and <offset>
      members of the returned structure, and <x>, <y>, and <z> specify the
      texel group number in the neighborhood.

      Each bit in the <mask> member of the returned structure corresponds to
      one of the texel groups in the 4x4x4 neighborhood.  That bit will be set
      if and only if any of the texels in the texel group is covered by the
      footprint.  The texel group (x,y,z) is considered to be covered if and
      only if the following logic computes true for <covered>:

        uint64_t mask = result.mask.x | (result.mask.y << 32);
        uint32_t bit = z * 16 + y * 4 + x;
        bool covered = (0 != ((mask >> bit) & 1));

      In most cases, the texel group sizes used by the footprint query will
      match the value passed to the query, as interpreted according to Table
      X.1.  However, in some cases, the footprint may be too large to be
      expressed as a collection of 8x8 or 4x4x4 set of texel groups using the
      requested granularity.  In this case, the implementation uses a texel
      group size that is larger than the requested granularity.  If a larger
      texel group size is used, the implementation will return the texel group
      size used in the <granularity> member of the footprint structure, which
      should also be interpreted according to Table X.1.  If the requested
      texel group size is used, the implementation will return zero in
      <granularity>.  The texel group size will only be increased by the
      implementation if anisotropic filtering is used.  If the texture and
      sampler objects used by the footprint query do not enable anisotropic
      texture filtering, the footprint query will always use the original
      requested granularity and return zero in the <granularity> member.

Errors

    None

New State

    None

New Implementation Dependent State

    None

Issues

    None, but please refer to issues in the GLSL extension specification.

Revision History

    Revision 2 (pknowles)
    - Add ES interactions.

    Revision 1 (clentini, pbrown)
    - Internal revisions.
