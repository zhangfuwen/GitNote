# ANGLE_texture_usage

Name

    ANGLE_texture_usage

Name Strings

    GL_ANGLE_texture_usage

Contributors

    Nicolas Capens, TransGaming
    Daniel Koch, TransGaming

Contact

    Daniel Koch, TransGaming (daniel 'at' transgaming.com)

Status

    Complete

Version

    Last Modified Date:  November 10, 2011
    Version:             2

Number

    OpenGL ES Extension #112 

Dependencies

    This extension is written against the OpenGL ES 2.0 Specification.

Overview

    In some implementations it is advantageous to know the expected
    usage of a texture before the backing storage for it is allocated.  
    This can help to inform the implementation's choice of format
    and type of memory used for the allocation. If the usage is not
    known in advance, the implementation essentially has to make a 
    guess as to how it will be used.  If it is later proven wrong,
    it may need to perform costly re-allocations and/or reformatting 
    of the texture data, resulting in reduced performance.

    This extension adds a texture usage flag that is specified via
    the TEXTURE_USAGE_ANGLE TexParameter.  This can be used to 
    indicate that the application knows that this texture will be 
    used for rendering.

IP Status

    No known IP claims.

New Procedures and Functions

    None

New Tokens

    Accepted as a value for <pname> for the TexParameter{if} and 
    TexParameter{if}v commands and for the <value> parameter of
    GetTexParameter{if}v: 

        TEXTURE_USAGE_ANGLE                     0x93A2

    Accepted as a value to <param> for the TexParameter{if} and 
    to <params> for the TexParameter{if}v commands with a <pname> of 
    TEXTURE_USAGE_ANGLE; returned as possible values for <data> when 
    GetTexParameter{if}v is queried with a <value> of TEXTURE_USAGE_ANGLE:

        NONE                                    0x0000
        FRAMEBUFFER_ATTACHMENT_ANGLE            0x93A3

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL ES Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    Add a new row to Table 3.10 (Texture parameters and their values):

    Name                | Type | Legal Values 
    ------------------------------------------------------------
    TEXTURE_USAGE_ANGLE | enum | NONE, FRAMEBUFFER_ATTACHMENT_ANGLE

    Add a new section 3.7.x (Texture Usage) before section 3.7.12 and 
    renumber the subsequent sections: 

    "3.7.x Texture Usage

    Texture usage can be specified via the TEXTURE_USAGE_ANGLE value
    for the <pname> argument to TexParameter{if}[v]. In order to take effect,
    the texture usage must be specified before the texture contents are
    defined either via TexImage2D or TexStorage2DEXT.

    The usage values can impact the layout and type of memory used for the 
    texture data. Specifying incorrect usage values may result in reduced
    functionality and/or significantly degraded performance.

    Possible values for <params> when <pname> is TEXTURE_USAGE_ANGLE are:

    NONE - the default. No particular usage has been specified and it is
        up to the implementation to determine the usage of the texture.
        Leaving the usage unspecified means that the implementation may 
        have to reallocate the texture data as the texture is used in 
        various ways.

    FRAMEBUFFER_ATTACHMENT_ANGLE - this texture will be attached to a 
        framebuffer object and used as a desination for rendering or blits."

    Modify section 3.7.12 (Texture State) and place the last 3 sentences
    with the following:

    "Next, there are the three sets of texture properties; each consists of
    the selected minification and magnification filters, the wrap modes for
    <s> and <t>, and the usage flags. In the initial state, the value assigned
    to TEXTURE_MIN_FILTER is NEAREST_MIPMAP_LINEAR, and the value for 
    TEXTURE_MAG_FILTER is LINEAR. <s> and <t> wrap modes are both set to
    REPEAT. The initial value for TEXTURE_USAGE_ANGLE is NONE."


Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special
Functions):

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and
State Requests)

    None

Dependencies on EXT_texture_storage

    If EXT_texture_storage is not supported, omit any references to 
    TexStorage2DEXT.

Errors

    If TexParameter{if} or TexParamter{if}v is called with a <pname>
    of TEXTURE_USAGE_ANGLE and the value of <param> or <params> is not
    NONE or FRAMEBUFFER_ATTACHMENT_ANGLE the error INVALID_VALUE is
    generated.

Usage Example

    /* create and bind texture */
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    /* specify texture parameters */
    glTexParameteri(GL_TEXTURE_2D, GL_*, ...);  /* as before */
    
    /* specify that we'll be rendering to the texture */
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_USAGE_ANGLE, GL_FRAMEBUFFER_ATTACHMENT_ANGLE);

    glTexStorage2DEXT(GL_TEXTURE_2D, levels, ...); // Allocation
    for(int level = 0; level < levels; ++level)
        glTexSubImage2D(GL_TEXTURE_2D, level, ...); // Initialisation

Issues

    1. Should there be a dynamic usage value?
   
       DISCUSSION: We could accept a dynamic flag to indicate that a texture will
       be updated frequently. We could map this to D3D9 dynamic textures. This would
       allow us to avoid creating temporary surfaces when updating the texture.
       However renderable textures cannot be dynamic in D3D9, which eliminates the 
       primary use case for this.  Furthermore, the memory usage of dynamic textures
       typically increases threefold when you lock it.

    2. Should the texture usage be an enum or a bitfield?

       UNRESOLVED.  Using a bitfield would allow combination of values to be specified.
       On the other hand, if combinations are really required, additional <pnames>
       could be added as necessary.  Querying a bitfield via the GetTexParameter command
       feels a bit odd.

    3. What should happen if TEXTURE_USAGE_ANGLE is set/changed after the texture
       contents have been specified?

       RESOLVED: It will have no effect. However, if the texture is redefined (for 
       example by TexImage2D) the new allocation will use the updated usage.
       GetTexParameter is used to query the value of the TEXTURE_USAGE_ANGLE
       state that was last set by TexParameter for the currently bound texture, or
       the default value if it has never been set. There is no way to determine the 
       usage that was in effect at the time the texture was defined.

Revision History

    Rev.    Date      Author     Changes
    ----  ----------- ---------  ----------------------------------------
      1   10 Nov 2011 dgkoch     Initial revision
      2   10 Nov 2011 dgkoch     Add overview
   

