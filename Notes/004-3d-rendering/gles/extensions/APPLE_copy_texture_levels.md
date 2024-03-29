# APPLE_copy_texture_levels

Name

    APPLE_copy_texture_levels
    
Name Strings

    GL_APPLE_copy_texture_levels

Contact

    Eric Sunalp, Apple Inc., (esunalp 'at' apple.com)
    
Contributors

    Alex Eddy, Apple
    Richard Schreyer, Apple
    Eric Sunalp, Apple
    Michael Swift, Apple

Status

    Complete
    
Version

    Last Modified Date:     November 29, 2011
    Revision:               2

Number

    OpenGL ES Extension #123

Dependencies

    OpenGL ES 1.1 or OpenGL ES 2.0 is required.
    
    EXT_texture_storage is required.
    
    This specification is written against the OpenGL ES 2.0.25 (Full Specification).

Overview

    This extension provides an efficient path for copying a contiguous subset of mipmap 
    levels from one texture to the matching subset of mipmap levels of another texture, 
    where matches are determined by the equality of a level's dimensions.
    
    This extension is dependent on the existence of the extension EXT_texture_storage.
    Immutable textures are used to guarantee that storage is allocated up front for the
    source and destination textures and that the internal formats of those textures are 
    sized the same.
    
    An efficient copy can be achieved by implementations because the internal storage 
    requirements are the same between textures and will remain unchanged when moving data. 
    It is expected that in all cases, moving levels from one texture to another is a 
    simple copy operation without any necessary conversion. This extension can be used as
    an alternative to TEXTURE_BASE_LEVEL. In some implementations, changing the value of
    TEXTURE_BASE_LEVEL can incur a costly re-allocation at runtime.
    
    Texture streaming is an expected use case for this extension. For example, a developer
    may want to stream in a larger base level for a given texture from a storage device. 
    To achieve this, a copy of the current mipmap levels are made into a destination 
    whose storage was specified to accommodate the source levels and the larger base 
    level. The efficiency of the copy without conversion allows for the smaller mipmap 
    levels to be in place while the larger base level is being read from the storage 
    device and uploaded.

New Tokens

    None
    
New Procedures and Functions

    void CopyTextureLevelsAPPLE(uint destinationTexture, uint sourceTexture,
                           int sourceBaseLevel, sizei sourceLevelCount);

New State

   None

New Implementation Dependent State

   None

Additions to Chapter 3 of the 2.0.25 Specification (Rasterization)

    -- Restate the first paragraph of section 3.7.2, Alternate Texture Image Specification 
       Commands

    Texture images may also be specified using image data taken directly from the  
    framebuffer or from a subset of levels of a given texture. Rectangular subregions of 
    existing texture images may also be respecified.

    -- Append to section 3.7.2, Alternate Texture Image Specification Commands

    The command
    
        void CopyTextureLevelsAPPLE(uint destinationTexture, uint sourceTexture,
                               int sourceBaseLevel, sizei sourceLevelCount);

    is used to specify a texture image by copying a contiguous subset of mipmap levels 
    from one texture to the matching subset of mipmap levels of another texture, where 
    matches are determined by the equality of a level's dimensions. An INVALID_OPERATION 
    is generated when the count and dimensions of the source texture levels don't exactly 
    match the count and dimensions of a subset of corresponding destination texture 
    levels.
    
    Both <sourceTexture> and <destinationTexture> specify the texture object names to act
    as the source and destination of the copy as apposed to operating on the currently 
    bound textures for a given set of texture units.
    
    It is a requirement that both <sourceTexture> and <destinationTexture> are immutable 
    textures and that they are both specified with the same sized internal format 
    enumeration. An INVALID_OPERATION is generated if either texture's 
    TEXTURE_IMMUTABLE_FORMAT_EXT is FALSE or if the sized internal formats don't match.
    It is a requirement that the <sourceTexture>'s target specification is the same as
    the <destinationTexture>'s target specification. If not, then an INVALID_OPERATION
    is generated.

    <sourceBaseLevel> and <sourceLevelCount> are used to specify the inclusive range of 
    mipmap levels to be copied from the source texture to the destination texture. 
    <sourceBaseLevel> can assume a value from zero to n-1, where n is the number of levels
    for which the source texture was specified. Anything outside of this range will result 
    in the generation of an INVALID_VALUE error. <sourceLevelCount> is added to 
    <sourceBaseLevel> to specify the range of levels to be copied to the destination. An 
    INVALID_VALUE is generated if that value is greater than the number of levels for
    which the source texture was specified. Additionally, <sourceLevelCount> must be 
    equal to or greater than one, or an INVALID_VALUE will be generated.

Errors

    The error INVALID_OPERATION is generated by CopyTextureLevelsAPPLE when the count and 
    dimensions of the source texture levels, from source base level to source base level 
    plus source level count, don't exactly match the count and dimensions of a subset of 
    matching destination texture levels.
    
    The error INVALID_OPERATION is generated by CopyTextureLevelsAPPLE if either the
    source texture or destination texture is a texture for which
    TEXTURE_IMMUTABLE_FORMAT_EXT is FALSE.
    
    The error INVALID_OPERATION is generated by CopyTextureLevelsAPPLE if the internal format
    of the source texture is different than that of the destination texture.
    
    The error INVALID_OPERATION is generated by CopyTextureLevelsAPPLE if the source and
    the destination target specification differ.
    
    The error INVALID_VALUE is generated by CopyTextureLevelsAPPLE if the value passed to
    the parameter <sourceTexture> or <destinationTexture> is zero.
    
    The error INVALID_VALUE is generated by CopyTextureLevelsAPPLE if the value passed to
    the parameter <sourceBaseLevel> is less than zero.
    
    The error INVALID_VALUE is generated by CopyTextureLevelsAPPLE if the value passed to
    the parameter <sourceBaseLevel> is greater than n-1, where n is the number of levels
    for which the source texture was specified.
    
    The error INVALID_VALUE is generated by CopyTextureLevelsAPPLE if the value passed to
    the parameter <sourceLevelCount>, is less than one, or when added to the parameter 
    <sourceBaseLevel>, is greater than n-1, where n is the number of levels for which the 
    source texture was specified.

Issues

    None
    
Revision History

    Revision 2, 2011/11/29 (Eric Sunalp)
        - Incorporate initial feedback.

    Revision 1, 2011/11/10 (Eric Sunalp)
        - Draft proposal.
