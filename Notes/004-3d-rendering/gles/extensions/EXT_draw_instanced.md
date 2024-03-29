# EXT_draw_instanced

Name

    EXT_draw_instanced

Name Strings

    GL_EXT_draw_instanced

Contact

    Michael Gold, NVIDIA Corporation (gold 'at' nvidia.com)

Status

    Shipping for GeForce 8 Series (November 2006)

Version

    Last Modified Date:  June 26, 2013
    Author Revision: 2.0

Number

    OpenGL Extension #327
    OpenGL ES Extension #157

Dependencies

    OpenGL 2.0 or OpenGL ES 2.0 is required.

    EXT_gpu_shader4 or NV_vertex_shader4 is required if the GL is not OpenGL ES 2.0.
    
    OES_element_index_uint affects the definition of this extension.

Overview

    This extension provides the means to render multiple instances of
    an object with a single draw call, and an "instance ID" variable
    which can be used by the vertex program to compute per-instance
    values, typically an object's transform.

New Tokens

    None

New Procedures and Functions

    void DrawArraysInstancedEXT(enum mode, int first, sizei count,
            sizei primcount);
    void DrawElementsInstancedEXT(enum mode, sizei count, enum type,
            const void *indices, sizei primcount);

Additions to Chapter 2 of the OpenGL 2.0 Specification
(OpenGL Operation)

    Modify section 2.8 (Vertex Arrays), p. 23

    (insert before the final paragraph, p. 30)

    The internal counter <instanceID> is a 32-bit integer value which
    may be read by a vertex program as <vertex.instance>, as described
    in section 2.X.3.2, or vertex shader as <gl_InstanceID>, as
    described in section 2.15.4.2.  The value of this counter is
    always zero, except as noted below.

    The command

        void DrawArraysInstancedEXT(enum mode, int first, sizei count,
                sizei primcount);

    behaves identically to DrawArrays except that <primcount>
    instances of the range of elements are executed and the value of
    <instanceID> advances for each iteration.  It has the same effect
    as:

        if (mode, count, or primcount is invalid)
            generate appropriate error
        else {
            for (i = 0; i < primcount; i++) {
                instanceID = i;
                DrawArrays(mode, first, count);
            }
            instanceID = 0;
        }

    The command

        void DrawElementsInstancedEXT(enum mode, sizei count, enum type,
                const void *indices, sizei primcount);

    behaves identically to DrawElements except that <primcount>
    instances of the set of elements are executed, and the value of
    <instanceID> advances for each iteration.  It has the same effect
    as:

        if (mode, count, primcount, or type is invalid )
            generate appropriate error
        else {
            for (int i = 0; i < primcount; i++) {
                instanceID = i;
                DrawElements(mode, count, type, indices);
            }
            instanceID = 0;
        }

Additions to Chapter 5 of the OpenGL 2.0 Specification
(Special Functions)

    The error INVALID_OPERATION is generated if DrawArraysInstancedEXT
    or DrawElementsInstancedEXT is called during display list
    compilation.

Dependencies on OpenGL ES 2.0

    If the GL is OpenGL ES 2.0, all references to vertex programs and display lists
    are deleted, and primcount is replaced by instanceCount in the function prototype
    and pseudocode.
    
    Add a new section in 2.10.5 called "Shader Inputs" between "Texture Access" and
    "Validation"

    "Besides having access to vertex attributes and uniform variables,
    vertex shaders can access the read-only built-in variable
    gl_InstanceIDEXT. The variable gl_InstanceIDEXT holds the integer
    index of the current primitive in an instanced draw call.  See also
    section 7.1 of the OpenGL ES Shading Language Specification."

    
    Additionally, the following is added to The OpenGL ES Shading Language Specification, 
    Version 1.00.17:

    "Including the following line in a shader can be used to control the
    language features described in this extension:

        #extension GL_EXT_draw_instanced : <behavior>

    where <behavior> is as specified in section 3.4.

    A new preprocessor #define is added to the OpenGL Shading Language:

        #define GL_EXT_draw_instanced 1

    Change Section 7.1 "Vertex Shader Special Variables"

    Add the following definitions to the list of built-in variable definitions:

        highp int gl_InstanceIDEXT; // read-only

    Add the following paragraph at the end of the section:

    The variable gl_InstanceIDEXT is available as a read-only variable
    from within vertex shaders and holds the integer index of the current
    primitive in an instanced draw call (DrawArraysInstancedEXT,
    DrawElementsInstancedEXT). If the current primitive does not come
    from an instanced draw call, the value of gl_InstanceIDEXT is zero."


Dependencies on NV_vertex_program4

    If NV_vertex_program4 is not supported and the GL is not OpenGL ES 2.0, 
    all references to vertex.instance are deleted.

Dependencies on EXT_gpu_shader4

    If EXT_gpu_shader4 is not supported and the GL is not OpenGL ES 2.0, 
    all references to gl_InstanceID are deleted.

Dependencies on OES_element_index_uint

    If OES_element_index_uint is not supported and the GL is OpenGL ES 2.0, 
    omit UNSIGNED_INT and uint from the description of DrawElementsInstancedEXT.

Errors

    INVALID_ENUM is generated by DrawElementsInstancedEXT if <type> is
    not one of UNSIGNED_BYTE, UNSIGNED_SHORT or UNSIGNED_INT.

    INVALID_VALUE is generated by DrawArraysInstancedEXT if <first> is
    less than zero.

Issues

  (1) Should instanceID be provided by this extension, or should it be
      provided by EXT_gpu_shader4, thus creating a dependence on that
      spec?

        Resolved: While this extension could stand alone, its utility
        would be limited without the additional functionality provided
        by EXT_gpu_shader4; also, the spec language is cleaner if
        EXT_gpu_shader4 assumes instanceID is always available, even
        if its value is always zero without this extension.
        
        For OpenGL ES 2.0: This extension does stand alone, introducing
        gl_InstanceID in GLSL-ES 1.00.

  (2) Should MultiDrawArrays and MultiDrawElements affect the value of
      instanceID?

        Resolved: No, this may cause implementation difficulties and
        is considered unlikely to provide any real benefit.

  (3) Should DrawArraysInstanced and DrawElementsInstanced be compiled
      into display lists?

        Resolved: No, calling these during display list compilation
        generate INVALID_OPERATION.

  (4) What is the maximum range of instances that gl_InstanceIDEXT can index 
      in an OpenGL ES 2.0 vertex shader?

        Resolved: According to the The OpenGL ES Shading Language 
        1.00.17 Specification, section 4.5.2 Precision Qualifiers, highp integer
        range is (-2^16, 2^16). So even though the DrawArraysInstancedEXT and 
        DrawElementsInstancedEXT take instanceCount as a 32-bit unsigned int,
        the maximum instance the gl_InstanceIDEXT variable can index is 
        2^16 - 1. If Instance count exceeds 2^16 - 1, it results in an undefined 
        value due to integer overflow and it is possible that gl_InstanceIDEXT 
        wraps, or that it does not.

Revision History

      Rev.    Date    Author    Changes
      ----  --------  --------  -----------------------------------------
      1.5   05/09/08  gold      Removed extraneous parameters to DrawArrays
                                and DrawElements in chapter 2 pseudocode

      2.0   06/26/13  benj      Add OpenGL ES 2.0 interactions

