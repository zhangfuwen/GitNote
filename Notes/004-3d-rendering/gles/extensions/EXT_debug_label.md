# EXT_debug_label

Name

    EXT_debug_label

Name Strings

    GL_EXT_debug_label

Contributors

    Seth Sowerby
    Benj Lipchak
    Jean-François Roy
    Charles Brissart

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status
    
    Complete

Version

    Date: October 7, 2013
    Revision: 4

Number

    OpenGL Extension #439
    OpenGL ES Extension #98

Dependencies
    
    Requires OpenGL ES 1.1.
    
    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

    OpenGL ES 1.1 affects the definition of this extension.

    OES_framebuffer_object affects the definition of this extension.
    
    OES_vertex_array_object affects the definition of this extension.

    EXT_occlusion_query_boolean affects the definition of this extension.

    EXT_separate_shader_objects affects the definition of this extension.
    
    OpenGL ES 3.0 affects the definition of this extension.
    
    OpenGL 3.3+ and 4.0+ affect the definition of this extension.

Overview

    This extension defines a mechanism for OpenGL and OpenGL ES applications to 
    label their objects (textures, buffers, shaders, etc.) with a descriptive 
    string. 
    
    When profiling or debugging such an application within a debugger or 
    profiler it is difficult to identify resources from their object names. 
    Even when the resource itself is viewed it can be problematic to 
    differentiate between similar resources. Attaching a label to an object         
    helps obviate this difficulty.
    
    The intended purpose of this is purely to improve the user experience 
    within OpenGL and OpenGL ES development tools.

New Procedures and Functions

    void LabelObjectEXT(enum type, uint object, sizei length, 
        const char *label);
    void GetObjectLabelEXT(enum type, uint object, sizei bufSize, 
        sizei *length, char *label);

New Tokens

    Accepted by the <type> parameter of LabelObjectEXT and 
    GetObjectLabelEXT:

        BUFFER_OBJECT_EXT                              0x9151
        SHADER_OBJECT_EXT                              0x8B48
        PROGRAM_OBJECT_EXT                             0x8B40
        VERTEX_ARRAY_OBJECT_EXT                        0x9154
        QUERY_OBJECT_EXT                               0x9153
        PROGRAM_PIPELINE_OBJECT_EXT                    0x8A4F
        
Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

    None

Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment
Operations and the Framebuffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    Add a new section titled Debug Labels

    Debug labels provide a method for annotating an object (texture, buffer, 
    shader, etc.) with a descriptive text label. These labels may then be used 
    by a tool such as a debugger or profiler to describe labeled objects. 
    
    The command
    
        void LabelObjectEXT(enum type, uint object, sizei length, 
            const char *label);
        
    labels the object <object> of type <type> with the label <label>. <length> 
    specifies the length of the string passed in <label>. If <label> is a null-
    terminated string then <length> should not include the terminator. If 
    <length> is 0 and <label> is non-null then <label> is assumed to be null-
    terminated. If <label> is NULL, any debug label is effectively removed from 
    <object>. 

    If <object> is not an object of type <type>, an INVALID_OPERATION error is 
    generated.
    
    A label is part of the state of the object to which it is associated. 
    The initial state of an object's label is NULL. Labels need not be unique.
    
    Values supported for <type> are: TEXTURE, FRAMEBUFFER, RENDERBUFFER,        
    BUFFER_OBJECT_EXT, SHADER_OBJECT_EXT, PROGRAM_OBJECT_EXT,
    VERTEX_ARRAY_OBJECT_EXT, QUERY_OBJECT_EXT, SAMPLER, TRANSFORM_FEEDBACK, and
    PROGRAM_PIPELINE_OBJECT_EXT.
    
Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State
Requests)

    Add a new section titled Debug Labels

    The command
    
        void GetObjectLabelEXT(enum type, uint object, sizei bufSize, 
            sizei *length, char *label);

    returns in <label> the string labeling the object <object>. The 
    string <label> will be null terminated. The actual number of characters 
    written into <label>, excluding the null terminator, is returned in 
    <length>. If <length> is NULL, no length is returned. The maximum number 
    of characters that may be written into <label>, including the null
    terminator, is specified by <bufSize>. If no debug label was specified for 
    the object <object> via LabelObjectEXT then <label> will contain an 
    empty string with null terminator and 0 will be returned in <length>. If 
    <label> is NULL and <length> is non-NULL then no string will be returned 
    and the length of the label will be returned in <length>. 
    
    If <object> is not an object of type <type>, an INVALID_OPERATION error is 
    generated.
    
    Values supported for <type> are: TEXTURE, FRAMEBUFFER, RENDERBUFFER,        
    BUFFER_OBJECT_EXT, SHADER_OBJECT_EXT, PROGRAM_OBJECT_EXT, 
    VERTEX_ARRAY_OBJECT_EXT, QUERY_OBJECT_EXT, SAMPLER, TRANSFORM_FEEDBACK, and
    PROGRAM_PIPELINE_OBJECT_EXT.
    
Interactions with OpenGL ES 1.1 and OES_framebuffer_object

    If the GL is OpenGL ES 1.1, mentions of SHADER_OBJECT_EXT and 
    PROGRAM_OBJECT_EXT as types accepted by LabelObjectEXT and 
    GetObjectLabelEXT are omitted.
    
    If OES_framebuffer_object is supported, FRAMEBUFFER and 
    RENDERBUFFER should be replaced by FRAMEBUFFER_OES and RENDERBUFFER_OES, 
    respectively. Otherwise they should be omitted.

Interactions with OpenGL ES 3.0 and OpenGL

    If the GL is not OpenGL ES 3.0 or OpenGL 3.3+, mentions of 
    SAMPLER as a type accepted by LabelObjectEXT and GetObjectLabelEXT are 
    omitted.

    If the GL is not OpenGL ES 3.0 or OpenGL 4.0+, mentions of 
    TRANSFORM_FEEDBACK as a type accepted by LabelObjectEXT and 
    GetObjectLabelEXT are omitted.

Interactions with OES_vertex_array_object

    If OES_vertex_array_object is not available, mentions of 
    VERTEX_ARRAY_OBJECT_EXT as a type accepted by LabelObjectEXT and 
    GetObjectLabelEXT is omitted.

Interactions with EXT_occlusion_query_boolean

    If EXT_occlusion_query_boolean is not available, mentions of 
    QUERY_OBJECT_EXT as a type accepted by LabelObjectEXT and GetObjectLabelEXT
    is omitted.

Interactions with EXT_separate_shader_objects

    If EXT_separate_shader_objects is not available, mentions of 
    PROGRAM_PIPELINE_OBJECT_EXT as a type accepted by LabelObjectEXT and 
    GetObjectLabelEXT is omitted.
    
Errors

    INVALID_OPERATION is generated by LabelObjectEXT or GetObjectLabelEXT 
    if the type of <object> does not match <type>.
    
    INVALID_ENUM error will be generated by LabelObjectEXT or 
    GetObjectLabelEXT if <type> is not one of the allowed object types.
    
    INVALID_VALUE is generated by LabelObjectEXT if <length> is less than
    zero.

    INVALID_VALUE is generated by GetObjectLabelEXT if <bufSize> is less than
    zero.

New State

    Add the following to Table 6.3 Buffer Object State:
    
                                                Initial
    Get Value          Type  Get Cmnd           Value    Description  Sec
    -----------------  ----  -----------------  -------  -----------  ---
    BUFFER_OBJECT_EXT  0*xc  GetObjectLabelEXT  empty    Debug label  5.X

    Add the following to Table 6.8 Textures (state per texture object):
    
                                          Initial
    Get Value  Type  Get Cmnd             Value    Description  Sec
    ---------  ----  -------------------  -------  -----------  ---
    TEXTURE    0*xc  GetObjectLabelEXT    empty    Debug label  5.X

    Add the following to Table 6.14 Shader Object State:
    
                                                Initial
    Get Value          Type  Get Cmnd           Value    Description  Sec
    -----------------  ----  -----------------  -------  -----------  ---
    SHADER_OBJECT_EXT  0*xc  GetObjectLabelEXT  empty    Debug label  5.X

    Add the following to Table 6.15 Program Object State:
    
                                                 Initial
    Get Value           Type  Get Cmnd           Value    Description  Sec
    ------------------  ----  -----------------  -------  -----------  ---
    PROGRAM_OBJECT_EXT  0*xc  GetObjectLabelEXT  empty    Debug label  5.X

    Add the following to Table 6.23 Renderbuffer State:
    
                                             Initial
    Get Value     Type  Get Cmnd             Value    Description  Sec
    ------------  ----  -------------------  -------  -----------  ---
    RENDERBUFFER  0*xc  GetObjectLabelEXT    empty    Debug label  5.X

    Add the following to Table 6.24 Framebuffer State:
    
                                            Initial
    Get Value    Type  Get Cmnd             Value    Description  Sec
    -----------  ----  -------------------  -------  -----------  ---
    FRAMEBUFFER  0*xc  GetObjectLabelEXT    empty    Debug label  5.X

    Add the following to Table 6.VAO Vertex Array Object State:
    
                                                      Initial
    Get Value                Type  Get Cmnd           Value    Description  Sec
    -----------------------  ----  -----------------  -------  -----------  ---
    VERTEX_ARRAY_OBJECT_EXT  0*xc  GetObjectLabelEXT  empty    Debug label  5.X

    Add the following to Table 6.QO Query State:
    
                                               Initial
    Get Value         Type  Get Cmnd           Value    Description  Sec
    ----------------  ----  -----------------  -------  -----------  ---
    QUERY_OBJECT_EXT  0*xc  GetObjectLabelEXT  empty    Debug label  5.X

    Add the following to Table 6.PPO Program Pipeline State:
    
                                                          Init.
    Get Value                    Type  Get Cmnd           Value  Description  Sec
    ---------------------------  ----  -----------------  -----  -----------  ---
    PROGRAM_PIPELINE_OBJECT_EXT  0*xc  GetObjectLabelEXT  empty  Debug label  5.X

    Add the following to Table 6.24 Transform Feedback State:

    Get Value                Type  Get Cmnd           Value    Description  Sec
    -----------------------  ----  -----------------  -------  -----------  ---
    TRANSFORM_FEEDBACK       0*xc  GetObjectLabelEXT  empty    Debug label  5.X

    Add the following to the Table 6.1 Textures (state per sampler object):

    Get Value                Type  Get Cmnd           Value    Description  Sec
    -----------------------  ----  -----------------  -------  -----------  ---
    SAMPLER                  0*xc  GetObjectLabelEXT  empty    Debug label  5.X


New Implementation Dependent State

    None

Issues

    (1) Should labels apply to the currently bound object for the specified 
    type instead of requiring the object name to be specified?
    
    Resolved: No.

    This would require shaders to be attached to a program which was then 
    linked and bound. This may not occur at the time the shader is created 
    and/or compiled. It might require significant work for developers to label 
    their shaders at the point of creation.
    
    (2) Should the extension accept FRAGMENT_SHADER & VERTEX_SHADER 
    instead of SHADER_OBJECT_EXT? ARRAY_BUFFER & ELEMENT_ARRAY_BUFFER instead 
    of BUFFER_OBJECT_EXT?
    
    Resolved: No.
    
    Specifying FRAGMENT_SHADER or VERTEX_SHADER to disambiguate the type of 
    shader would be necessary only if the resolution to issue #1 were yes,
    in which case one would label a shader attached to the currently bound 
    program without explicitly specifying the shader's name.  The same applies
    to buffer objects bound to the current vertex attribute array target versus
    the current element array target.

    (3) Should the extension support printf-style formatting?

    Resolved: No.

    Providing printf-style formatting would impose a much greater burden on the 
    extension in terms of error checking the format string and arguments. 
    Likely all languages capable of calling OpenGL ES have convenient 
    capabilities for formatting strings so it is acceptable to rely on those.
    
    (4) Should labeling a non-existent object effectively create the object?
    
    Resolved: No, since some objects require more information (e.g. a texture
    target) to be properly initialized.
    
Revision History

    Date 06/15/2011
    Revision: 1
       - draft proposal

    Date 07/22/2011
    Revision: 2
       - rename APPLE to EXT, update token names and values

    Date 10/18/2012
    Revision: 3
       - Add OpenGL ES 3.0 interactions: transform feedback and sampler objects
         
    Date 10/07/2013
    Revision: 4
       - Add support for desktop OpenGL
