# EXT_frag_depth

Name

    EXT_frag_depth

Name Strings

    GL_EXT_frag_depth

Contributors

    Maurice Ribble
    Robert Simpson
    Jeff Leger
    Bruce Merry
    Acorn Pooley

Contact

    Maurice Ribble (mribble 'at' qualcomm.com)

Notice

    None

Status

    Draft

Version

    Date: July 21, 2010

Number

    OpenGL ES Extension #86

Dependencies

    OpenGL ES 2.0 is required.
    
    This extension is written against the OpenGL ES 2.0 specification, and
    the OpenGL ES Shading Language 1.0.17 specification.
    
    OES_fragment_precision_high affects the definitions of this extension.

Overview

    This extension adds the ability to set the depth value of a fragment from
    within the fragment shader.  Then this per-fragment depth value is used
    for depth testing.  This extension adds a built-in GLSL fragment shader 
    special varible to set the depth value.
    
    Much graphics hardware has the ability to do early depth testing before the
    fragment shader.  On such hardware there may be a performance penality for
    using this feature so use this feature only when needed.

Issues

    (1) Should the GLSL keyword be gl_FragDepth or gl_FragDepthEXT?
    
    RESOLVED: OpenGL has discussed this in the past and the agreement was that
    we should use gl_FragDepthEXT.
    
    (2) What should the precission qualifier be for gl_FragDepthEXT?
    
    RESOLVED: If the OES_fragment_precision_high is supported then highp is
    used, but if OES_fragment_precision_high not supported then mediump is 
    used.

New Procedures and Functions

    None

New Tokens

    None

New Keywords

    gl_FragDepthEXT

New Built-in Functions

    None

New Macro Definitions

    #define GL_EXT_frag_depth 1

Additions to Appendix A.3 Invariance Rules

    Rule 4: All fragment shaders that either conditionally or unconditionally
    assign gl_FragCoord.z to gl_FragDepthEXT are depth-invariant with respect
    to each other, for those fragments where the assignment to gl_FragDepthEXT
    actually is done.

Additions to Chapter 7 of the OpenGL ES Shading Language specification:

    Make the following changes to section 7.2 (Fragment Shader Special 
    Varibles).

    Replace the last sentence in the first paragraph with this:

    "Fragment shaders output values to the OpenGL ES pipeline using the 
    built-in variables gl_FragColor, gl_FragData, and gl_FragDepthEXT, unless the
    discard keyword is executed."
    
    Add this between the first and second paragraphs:
    
    "The built-in varible gl_FragDepthEXT is optional, and must be enabled by

    #extension GL_EXT_frag_depth : enable

    before being used."
   
    Replace the first sentence in the second paragraph with this:
    
    "It is not a requirement for the fragment shader to write to gl_FragColor,
    gl_FragData, or gl_FragDepthEXT."

    Add this paragraph after the paragraph that starts with "Writing to 
    gl_FragColor":

    "Writing to gl_FragDepthEXT will establish the depth value for the fragment 
    being processed. If MSAA is enabled, the depth value is copied to all
    samples corresponding to the fragment. If depth buffering is enabled, and
    no shader writes gl_FragDepthEXT, then the fixed function value for depth 
    will be used as the fragment's depth value. If a shader statically assigns
    a value to gl_FragDepthEXT, and there is an execution path through the 
    shader that does not set gl_FragDepthEXT, then the value of the fragment's
    depth may be undefined for executions of the shader that take that path. 
    That is, if the set of linked fragment shaders statically contain a write 
    to gl_FragDepthEXT, then it is responsible for always writing it."

    Replace the paragraph that starts with "If a shader executes the discard" 
    with this:
    
    "If a shader executes the discard keyword, the fragment is discarded, and
    the values of any user-defined fragment outputs, gl_FragDepthEXT, 
    gl_FragColor, and gl_FragData become irrelevant."

    Replace the last sentence of the 9th paragraph with the following:

    The z component is the depth value that would be used for the fragment's
    depth if the shader contained no writes to gl_FragDepthEXT.  This is useful
    for invariance if a shader conditionally computes gl_FragDepthEXT but 
    otherwise wants the fixed functionality fragment depth.
    
    Add this to the list of built-in varibles:
    
    If OES_fragment_precision_high is supported add this:
      "highp float gl_FragDepthEXT;"
    otherwise add this:
      "mediump float gl_FragDepthEXT;"

New State

    None

Revision History

    6/14/2010  Created.
    6/14/2010  Added language to cover MSAA.
    6/15/2010  Fixed some typos.
    7/2/2010   Fixed issues from Bruce.
                Added wording to Appendix A.3.
                Added issues 1 and 2.
    7/8/2010   Changed from OES to EXT.
               Various updates from NV version of this extension.
    7/21/2010  Resolved issues 1 and 2.
                
