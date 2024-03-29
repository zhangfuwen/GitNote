# EXT_protected_textures

Name

    EXT_protected_textures

Name Strings

    GL_EXT_protected_textures

Contributors

    Maurice Ribble
    Mohan Maiya
    Craig Donner
    Alex Wong
    Jan-Harald Fredriksen
    Daniel Koch
    Prabindh Sundareson
    Jesse Hall
    Ray Smith
    Jonathan Putsman

Contact

    Jeff Leger (jleger 'at' qti.qualcomm.com)

IP Status

    No known IP claims.

Status

    Complete

Version

    May 17, 2017

Number

    OpenGL ES Extension #256

Dependencies

    OpenGL ES 3.0 is required.

    This extension requires the EGL_EXT_protected_content extension or
    similar.

    This extension is written against the OpenGL ES 3.2 specification.
    
    Interacts with EXT_sparse_texture.

Overview

    This extension requires another extension like EGL_EXT_protected_content to
    have created a protected context.  A protected context enables the
    driver to put the GPU into a protected mode where it is able to operate on
    protected surfaces.

    This extension enables allocating standard GL textures as protected
    surfaces. Previously these textures would have had to have been created as
    special EGL surfaces.  This allows use-cases such as depth, stencil, or
    mipmapped textures to be supported as destinations for rendering within
    a protected context.

    An explanation of undefined behavior in this extension:  Several places
    in this extension mention undefined behavior can result, which can
    include program termination.  The reason for this is because one way
    to handle protected content is by using a protected virtual to physical
    memory translation layer.  With this sort of solution a system may generate
    read or write faults when a non-protected context tries to access the
    buffer.  Depending on the system these faults might be ignored or they might
    cause process termination.  This undefined behavior should not include
    actually allowing copying any protected content to a non-protected surface.

    This extension does not guarantee that the implementation abides by a
    system's digital rights management requirements.  It must be verified beyond
    the existence of this extension that the implementation of this extension is
    trustworthy according to the requirements of a content protection system.

New Procedures and Functions

    None

New Tokens

    Returned by GetIntegerv when <pname> is CONTEXT_FLAGS:

        GL_CONTEXT_FLAG_PROTECTED_CONTENT_BIT_EXT       0x00000010

    Accepted as a value for <pname> for the TexParameter{if} and
    TexParameter{if}v commands and for the <value> parameter of
    GetTexParameter{if}v:

        GL_TEXTURE_PROTECTED_EXT                        0x8BFA

    Accepted as a value to <param> for the TexParameter{if} and
    to <params> for the TexParameter{if}v commands with a <pname> of
    TEXTURE_PROTECTED_EXT; returned as possible values for <data> when
    GetTexParameter{if}v is queried with a <value> of TEXTURE_PROTECTED_EXT:

        FALSE                                           0x0
        TRUE                                            0x1

Add the following to the end of section 8.18 (Immutable-Format Texture Images)
in the OpenGL ES 3.2 Specification:

    To use protected textures a context must be a protected context.
    To check if your context supports protected content you can
    query the context by calling GetIntegerv with pname CONTEXT_FLAGS, as
    described in section 20.2.

    Texture usage can be specified via the TEXTURE_PROTECTED_EXT value
    for the <pname> argument to TexParameter{if}[v]. In order to take effect,
    the texture usage must be specified before the texture contents are
    defined by TexStorage*.

    Possible values for <params> when <pname> is TEXTURE_PROTECTED_EXT are:

    FALSE - the default. The texture is not protected.

    TRUE - the texture is protected.

    The definition of protected and non-protected access is up to the
    implementation and is out of scope of this specification.  To read/write a
    protected surface, it is required that the context also be protected.

    CONTEXT_FLAG_PROTECTED_CONTENT_BIT_EXT is set when the context is
    created in protected mode with an extension such as
    EGL_EXT_protected_context.

Add a new section "Protected Content" to Chapter 2 "OpenGL ES Operation":

    If the context is protected, the pipeline stages are executed in the
    following manner:
    - The fragment stage is always protected
    - For all other stages, it is implementation defined whether that stage is
    protected or not protected.

    Permitted operations in protected and not protected mode are out of the
    scope of this specification. Refer to EGL_EXT_protected_context, if
    applicable, for more information.

Add a new row to Table 8.19 (Texture parameters and their values):

    Name                  | Type | Legal Values
    ------------------------------------------------------------
    TEXTURE_PROTECTED_EXT | bool | FALSE, TRUE

Errors

    If TexParameter{if} or TexParamter{if}v is called with a <pname>
    of TEXTURE_PROTECTED_EXT and the value of <param> or <params> is not
    TRUE or FALSE the error INVALID_VALUE is generated.

    [[ The following is only added if EXT_sparse_texture is supported. ]]
    
    Add to the errors that may be generated by TexStorage*:

        An INVALID_OPERATION error is generated if the texture's
        TEXTURE_SPARSE_EXT parameter is TRUE and the value of its
        TEXTURE_PROTECTED_EXT parameter is TRUE.  

Issues
    1) Should this work with all texture functions or only the new texture
    storage allocations?

    RESOLVED - Only supporting a new texture storage allocation method is much
    simpler than trying to alter every texture-related entry-point.


    2) Some paths like TexSubImage may have GPU and CPU paths.  Should those
    types of operations be supported in this extension.

    PROPOSED - Yes.  While protected surfaces can't be updated with
    non-secure GPU/CPU operations, an implementation is expected to either use
    trusted GPU or CPU operations to accomplish this.

    3) What target should all the texture bind and parameter setting API calls
    use?

    RESOLVED - They will use the non *_PROTECTED targets.  This was done to
    reduce the amount of code change in apps.  The only calls using *_PROTECTED
    targets are the texStorage allocation calls.

    4) What happens if readPixels is performed on a protected surface?

    RESOLVED - Results will not actually get the surface data, but
    otherwise the results are undefined up to and including app termination.

    5) Can you create an EGLImage from a protected texture?

    RESOLVED - Yes, but only if the EGLImage is created in a protected context.

    6) Should all textures on a protected context be protected by default?

    RESOLVED - No, several implementations have limited amounts of protected
    memory so the API will require opting into protected memory.

    7) How should protected memory be exposed?

    RESOLVED - Options discussed where target *_PROTECTED, using
    TexStorage*WithFlags, and adding a new texture parameter.  The group decided
    to add a new texture parameter.

    8) If an FBO attachment is protected, must all of the attachments be
    protected?

    RESOLVED - Yes, if any of the FBO attachments are protected then they all
    must be protected or the results are undefined.  If this is being used
    in conjunction with the EGL_EXT_protected_content extension that extension
    states all outputs must be protected for a protected context, and that
    inputs can be mixed.

    9) Are occlusion, timer, and other types of queries allowed when using the
    the EGL_EXT_protected_content extension?

    RESOLVED - No, these will result in undefined behavior with this extension.
    These features require writing to a buffer and a protected context can only
    write to a protected surface.  There are no protected buffers so this isn't
    possible.  Even if there were protected buffers that data wouldn't be
    visible on the CPU.

    10) What is the interaction between EXT_protected_textures and
    EXT_sparse_texture?

    RESOLVED - It is forbidden to create a texture which is both protected and
    sparse.   This is problematic on some platforms and there is no known
    compelling use case.

Revision History

    Rev.    Date     Author    Changes
    ----  --------  --------  ----------------------------------------------
     1    03/07/16  mribble   Initial draft.
     2    03/10/16  mribble   Cleanup.
     3    03/18/16  mribble   Fix issues brought up by Khronos group.
     4    03/25/16  mribble   Changed to tex parameter method.  Other cleanup.
     5    03/30/16  mribble   Added issues 8 and 9.  Other cleanup.
     6    04/08/16  rsmith    Added section on Protected Content defining the
                              protection state of each pipeline stage, as
                              required by EGL_EXT_protected_content.
     7    04/10/16  mribble   Minor cleanup.
     8    04/11/16  mribble   Clarify issue 9.
     9    04/11/16  Jon Leech Add missing _EXT suffix to context flag.
     10   05/17/17  jleger    Add issue 10 and the corresponding error.
