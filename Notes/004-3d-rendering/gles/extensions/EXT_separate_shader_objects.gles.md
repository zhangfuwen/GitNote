# EXT_separate_shader_objects

Name

    EXT_separate_shader_objects

Name Strings

    GL_EXT_separate_shader_objects

Contributors

    Contributors to ARB_separate_shader_objects and 
    ARB_explicit_attrib_location desktop OpenGL extensions from which this 
    extension borrows heavily

Contact

    Benj Lipchak, Apple (lipchak 'at' apple.com)

Status

    Complete

    NOTE: there is an unrelated OpenGL extension also named
    "GL_EXT_separate_shader_objects", found in the OpenGL extension registry
    at http://www.opengl.org/registry/ . These two extensions have similar
    purposes, but completely different interfaces.

Version

    Date: November 08, 2013
    Revision: 7

Number

    OpenGL ES Extension #101

Dependencies

    Requires OpenGL ES 2.0.

    Written based on the wording of the OpenGL ES 2.0.25 Full Specification
    (November 2, 2010).

    Written based on the wording of The OpenGL ES Shading Language 1.0.17
    Specification (May 12, 2009).
    
    OpenGL ES 3.0 affects the definition of this extension.

    NV_non_square_matrices affects the definition of this extension.

Overview

    This extension is a subset of ARB_separate_shader_objects appropriate for
    OpenGL ES, and also tacks on ARB_explicit_attrib_location functionality.

    Conventional GLSL requires multiple shader stages (vertex and fragment) to
    be linked into a single monolithic program object to specify a GLSL shader 
    for each stage.

    While GLSL's monolithic approach has some advantages for optimizing shaders 
    as a unit that span multiple stages, GPU hardware supports a more flexible 
    mix-and-match approach to specifying shaders independently for these 
    different shader stages.  Many developers build their shader content around
    the mix-and-match approach where they can use a single vertex shader with 
    multiple fragment shaders (or vice versa).

    This extension adopts a "mix-and-match" shader stage model for GLSL 
    allowing multiple different GLSL program objects to be bound at once each 
    to an individual rendering pipeline stage independently of other stage 
    bindings. This allows program objects to contain only the shader stages 
    that best suit the application's needs.

    This extension introduces the program pipeline object that serves as a 
    container for the program bound to any particular rendering stage.  It can 
    be bound, unbound, and rebound to simply save and restore the complete 
    shader stage to program object bindings.  Like framebuffer
    and vertex array objects, program pipeline objects are "container"
    objects that are not shared between contexts.

    To bind a program object to a specific shader stage or set of stages, 
    UseProgramStagesEXT is used.  The VERTEX_SHADER_BIT_EXT and 
    FRAGMENT_SHADER_BIT_EXT tokens refer to the conventional vertex and 
    fragment stages, respectively. ActiveShaderProgramEXT specifies the 
    program that Uniform* commands will update.

    While ActiveShaderProgramEXT provides a selector for setting and querying 
    uniform values of a program object with the conventional Uniform* commands,
    the ProgramUniform* commands provide a selector-free way to modify uniforms
    of a GLSL program object without an explicit bind. This selector-free model
    reduces API overhead and provides a cleaner interface for applications.

    Separate linking creates the possibility that certain output varyings of a 
    shader may go unread by the subsequent shader input varyings. In this 
    case, the output varyings are simply ignored. It is also possible input 
    varyings from a shader may not be written as output varyings of a preceding
    shader. In this case, the unwritten input varying values are undefined.

    This extension also introduces a layout location qualifier to GLSL to pre-
    assign attribute and varying locations to shader variables.  This allows 
    applications to globally assign a particular semantic meaning, such as 
    diffuse color or vertex normal, to a particular attribute and/or varying 
    location without knowing how that variable will be named in any particular 
    shader.
    
New Procedures and Functions

    void UseProgramStagesEXT(uint pipeline, bitfield stages,
                             uint program);

    void ActiveShaderProgramEXT(uint pipeline, uint program);

    uint CreateShaderProgramvEXT(enum type, sizei count,
                                 const char **strings);

    void BindProgramPipelineEXT(uint pipeline);

    void DeleteProgramPipelinesEXT(sizei n, const uint *pipelines);

    void GenProgramPipelinesEXT(sizei n, uint *pipelines);

    boolean IsProgramPipelineEXT(uint pipeline);

    void ProgramParameteriEXT(uint program, enum pname, int value);

    void GetProgramPipelineivEXT(uint pipeline, enum pname, int *params);

    void ProgramUniform1iEXT(uint program, int location,
                             int x);
    void ProgramUniform2iEXT(uint program, int location,
                             int x, int y);
    void ProgramUniform3iEXT(uint program, int location,
                             int x, int y, int z);
    void ProgramUniform4iEXT(uint program, int location,
                             int x, int y, int z, int w);

    void ProgramUniform1fEXT(uint program, int location,
                             float x);
    void ProgramUniform2fEXT(uint program, int location,
                             float x, float y);
    void ProgramUniform3fEXT(uint program, int location,
                             float x, float y, float z);
    void ProgramUniform4fEXT(uint program, int location,
                             float x, float y, float z, float w);

    void ProgramUniform1uiEXT(uint program, int location,
                              uint x);
    void ProgramUniform2uiEXT(uint program, int location,
                              uint x, uint y);
    void ProgramUniform3uiEXT(uint program, int location,
                              uint x, uint y, uint z);
    void ProgramUniform4uiEXT(uint program, int location,
                              uint x, uint y, uint z, uint w);

    void ProgramUniform1ivEXT(uint program, int location,
                              sizei count, const int *value);
    void ProgramUniform2ivEXT(uint program, int location,
                              sizei count, const int *value);
    void ProgramUniform3ivEXT(uint program, int location,
                              sizei count, const int *value);
    void ProgramUniform4ivEXT(uint program, int location,
                              sizei count, const int *value);

    void ProgramUniform1fvEXT(uint program, int location,
                              sizei count, const float *value);
    void ProgramUniform2fvEXT(uint program, int location,
                              sizei count, const float *value);
    void ProgramUniform3fvEXT(uint program, int location,
                              sizei count, const float *value);
    void ProgramUniform4fvEXT(uint program, int location,
                              sizei count, const float *value);

    void ProgramUniform1uivEXT(uint program, int location,
                               sizei count, const uint *value);
    void ProgramUniform2uivEXT(uint program, int location,
                               sizei count, const uint *value);
    void ProgramUniform3uivEXT(uint program, int location,
                               sizei count, const uint *value);
    void ProgramUniform4uivEXT(uint program, int location,
                               sizei count, const uint *value);

    void ProgramUniformMatrix2fvEXT(uint program, int location,
                                    sizei count, boolean transpose,
                                    const float *value);
    void ProgramUniformMatrix3fvEXT(uint program, int location,
                                    sizei count, boolean transpose,
                                    const float *value);
    void ProgramUniformMatrix4fvEXT(uint program, int location,
                                    sizei count, boolean transpose,
                                    const float *value);
    void ProgramUniformMatrix2x3fvEXT(uint program, int location,
                                     sizei count, boolean transpose,
                                     const float *value);
    void ProgramUniformMatrix3x2fvEXT(uint program, int location,
                                     sizei count, boolean transpose,
                                     const float *value);
    void ProgramUniformMatrix2x4fvEXT(uint program, int location,
                                     sizei count, boolean transpose,
                                     const float *value);
    void ProgramUniformMatrix4x2fvEXT(uint program, int location,
                                     sizei count, boolean transpose,
                                     const float *value);
    void ProgramUniformMatrix3x4fvEXT(uint program, int location,
                                     sizei count, boolean transpose,
                                     const float *value);
    void ProgramUniformMatrix4x3fvEXT(uint program, int location,
                                     sizei count, boolean transpose,
                                     const float *value);


   void ValidateProgramPipelineEXT(uint pipeline);

   void GetProgramPipelineInfoLogEXT(uint pipeline, sizei bufSize,
                                     sizei *length, char *infoLog);

New Tokens

    Accepted by <stages> parameter to UseProgramStagesEXT:

        VERTEX_SHADER_BIT_EXT                0x00000001
        FRAGMENT_SHADER_BIT_EXT              0x00000002
        ALL_SHADER_BITS_EXT                  0xFFFFFFFF

    Accepted by the <pname> parameter of ProgramParameteriEXT and
    GetProgramiv:

        PROGRAM_SEPARABLE_EXT                0x8258

    Accepted by <type> parameter to GetProgramPipelineivEXT:

        ACTIVE_PROGRAM_EXT                   0x8259

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:

        PROGRAM_PIPELINE_BINDING_EXT         0x825A

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation):

 -- Section 2.10 "Vertex Shaders" (page 26)

    Modify the fourth and fifth paragraphs:

    "To use a vertex shader, shader source code is first loaded into a shader 
    object and then compiled. Alternatively, pre-compiled shader binary code 
    may be directly loaded into a shader object. An OpenGL ES implementation 
    must support one of these methods for loading shaders. If the boolean value
    SHADER_COMPILER is TRUE, then the shader compiler is supported. If the 
    integer value NUM_SHADER_BINARY_FORMATS is greater than zero, then shader 
    binary loading is supported.
    
    A shader object corresponds to a stage in the rendering pipeline referred 
    to as its shader stage or type. A vertex shader object is attached to a 
    program object. The program object is then linked, which generates 
    executable code from the compiled shader object(s) attached to the program. 
    When program objects are bound to a shader stage, they become the current 
    program object for that stage. When the current program object for the 
    vertex stage includes a vertex shader, it is considered the active program 
    object for the vertex stage. The current program object for all stages may
    be set at once using a single unified program object, or the current 
    program object may be set for each stage individually using a separable 
    program object where different separable program objects may be current for
    other stages. The set of separable program objects current for all stages
    are collected in a program pipeline object that must be bound for use.
    When a linked program object is made active for the vertex stage, the 
    executable code for the vertex shaders it contains is used to process 
    vertices."
    
    Modify the last sentence in the sixth paragraph:

    "... A single program object can contain either a vertex shader, a fragment
    shader, or both."

    Modify the seventh paragraph:

    "When the program object currently in use for the vertex stage
    includes a vertex shader, its vertex shader is considered active and
    is used to process vertices. If the current vertex stage program
    object has no vertex shader or no program object is current for the
    vertex stage, the results of vertex shader execution are undefined."

 -- Section 2.10.3 "Program Objects" (page 30) 

    Modify the description of linking failures following the description of
    LinkProgram to read:
    
    "... if <program> is not separable and does not contain both a vertex and a
    fragment shader, ..."

    Modify second paragraph, p. 31:

    "If a program has been successfully linked by LinkProgram, it can be made
    part of the current rendering state for all shader stages with the command

        void UseProgram(uint program);

    If <program> is non-zero, this command will make <program> the current
    program object. This will install executable code as part of the current
    rendering state for each shader stage present when the program was last
    successfully linked. If UseProgram is called with <program> set to zero,
    then there is no current program object. If <program> has not been
    successfully linked, the error INVALID_OPERATION is generated and the
    current rendering state is not modified."

    Insert paragraph immediately after preceding paragraph:

    "The executable code for an individual shader stage is taken from the
    current program for that stage. If there is a current program object
    established by UseProgram, that program is considered current for all
    stages. Otherwise, if there is a bound program pipeline object (section
    2.10.PPO), the program bound to the appropriate stage of the pipeline
    object is considered current. If there is no current program object or
    bound program pipeline object, no program is current for any stage. The
    current program for a stage is considered active if it contains executable
    code for that stage; otherwise, no program is considered active for that
    stage. If there is no active program for the vertex or fragment shader
    stages, the results of vertex and/or fragment processing will be undefined. 
    However, this is not an error."

    Modify fourth and fifth paragraphs, p. 31:

    "If a program object that is active for any shader stage is re-linked
    successfully, the LinkProgram command will install the generated
    executable code as part of the current rendering state for all shader
    stages where the program is active. Additionally, the newly generated
    executable code is made part of the state of any program pipeline for all
    stages where the program is attached.

    If a program object that is active for any shader stage is re-linked
    unsuccessfully, the link status will be set to FALSE, but existing
    executables and associated state will remain part of the current rendering
    state until a subsequent call to UseProgram, UseProgramStagesEXT, or
    BindProgramPipelineEXT removes them from use. If such a program is 
    attached to any program pipeline object, the existing executables and 
    associated state will remain part of the program pipeline object until a 
    subsequent call to UseProgramStagesEXT removes them from use. An 
    unsuccessfully linked program may not be made part of the current rendering 
    state by UseProgram or added to program pipeline objects by 
    UseProgramStagesEXT until it is successfully re-linked. If such a program 
    was attached to a program pipeline at the time of a failed link, its 
    existing executable may still be made part of the current rendering state 
    indirectly by BindProgramPipelineEXT."

    Insert prior to the description of DeleteProgram, p. 31:

    "Program parameters control aspects of how the program is linked,
    executed, or stored. To set a program parameter, call

        void ProgramParameteriEXT(uint program, enum pname, int value);

    <pname> identifies which parameter to set for program object
    <program>. <value> holds the value being set.
    
    If <pname> is PROGRAM_SEPARABLE_EXT, <value> must be TRUE or FALSE
    and indicates whether the <program> can be bound for individual
    pipeline stages via UseProgramStagesEXT after it is next linked."
        
    Modify the last paragraph of the section, p. 31:

    "If <program> is not current for any GL context, is not the active
    program for any program pipeline object, and is not the current
    program for any stage of any program pipeline object, it is deleted
    immediately. Otherwise, <program> is flagged ..."

    Insert at the end of the section, p. 31:

    "The command

        uint CreateShaderProgramvEXT(enum type, sizei count,
                                     const char **strings);

    creates a stand-alone program from an array of null-terminated source 
    code strings for a single shader type.  CreateShaderProgramvEXT
    is equivalent to the following command sequence:

        const uint shader = CreateShader(type);
        if (shader) {
            ShaderSource(shader, count, strings, NULL);
            CompileShader(shader);
            const uint program = CreateProgram();
            if (program) {
                int compiled = FALSE;
                GetShaderiv(shader, COMPILE_STATUS, &compiled);
                ProgramParameteriEXT(program, PROGRAM_SEPARABLE_EXT, TRUE);
                if (compiled) {
                    AttachShader(program, shader);
                    LinkProgram(program);
                    DetachShader(program, shader);
                }
                append-shader-info-log-to-program-info-log
            }
            DeleteShader(shader);
            return program;
        } else {
            return 0;
        }

    The program may not actually link if the output variables in the
    shader attached to the final stage of the linked program take up
    too many locations. If this situation arises, the info log may
    explain this.

    Because no shader is returned by CreateShaderProgramvEXT and the
    shader that is created is deleted in the course of the command
    sequence, the info log of the shader object is copied to the program
    so the shader's failed info log for the failed compilation is
    accessible to the application."

 -- Add new section 2.10.PPO "Program Pipeline Objects" after 2.10.3
    "Program Objects"

    "Instead of packaging all shader stages into a single program object,
    shader types might be contained in multiple program objects each
    consisting of part of the complete pipeline. A program object may
    even contain only a single shader stage. This facilitates greater
    flexibility when combining different shaders in various ways without
    requiring a program object for each combination.

    Program bindings associating program objects with shader types are
    collected to form a program pipeline object.

    The command

        void GenProgramPipelinesEXT(sizei n, uint *pipelines);

    returns <n> previously unused program pipeline object names in
    <pipelines>. These names are marked as used, for the purposes of
    GenProgramPipelinesEXT only, but they acquire state only when they are
    first bound.

    Program pipeline objects are deleted by calling

        void DeleteProgramPipelinesEXT(sizei n, const uint *pipelines);
    
    <pipelines> contains <n> names of program pipeline objects to be
    deleted. Once a program pipeline object is deleted, it has no
    contents and its name becomes unused. If an object that is currently
    bound is deleted, the binding for that object reverts to zero and no
    program pipeline object becomes current. Unused names in <pipelines>
    are silently ignored, as is the value zero.

    A program pipeline object is created by binding a name returned by
    GenProgramPipelinesEXT with the command

        void BindProgramPipelineEXT(uint pipeline);

    <pipeline> is the program pipeline object name. The resulting program
    pipeline object is a new state vector, comprising all the state and with
    the same initial values listed in table 6.PPO.

    BindProgramPipelineEXT may also be used to bind an existing program
    pipeline object. If the bind is successful, no change is made to
    the state of the bound program pipeline object, and any previous
    binding is broken. If BindProgramPipelineEXT is called with <pipeline>
    set to zero, then there is no current program pipeline object.

    If no current program object has been established by UseProgram, the
    program objects used for each shader stage and for uniform updates are
    taken from the bound program pipeline object, if any. If there is a
    current program object established by UseProgram, the bound program
    pipeline object has no effect on rendering or uniform updates. When a
    bound program pipeline object is used for rendering, individual shader
    executables are taken from its program objects as described in the
    discussion of UseProgram in section 2.10.3.

    BindProgramPipelineEXT fails and an INVALID_OPERATION error is
    generated if <pipeline> is not zero or a name returned from a
    previous call to GenProgramPipelinesEXT, or if such a name has since
    been deleted with DeleteProgramPipelinesEXT.

    The executables in a program object associated with one or more
    shader stages can be made part of the program pipeline state for
    those shader stages with the command:

       void UseProgramStagesEXT(uint pipeline, bitfield stages,
                                uint program);

    where <pipeline> is the program pipeline object to be updated,
    <stages> is the bitwise OR of accepted constants representing
    shader stages, and <program> is the program object from which the
    executables are taken. The bits set in <stages> indicate the program
    stages for which the program object named by <program> becomes
    current. These stages may include vertex or fragment indicated by 
    VERTEX_SHADER_BIT_EXT or FRAGMENT_SHADER_BIT_EXT respectively. The 
    constant ALL_SHADER_BITS_EXT indicates <program> is to be made current 
    for all shader stages. If <program> refers to a program object with a valid
    shader attached for an indicated shader stage, this call installs
    the executable code for that stage in the indicated program pipeline
    object state. If UseProgramStagesEXT is called with <program> set to
    zero or with a program object that contains no executable code for the
    given stages, it is as if the pipeline object has no programmable stage
    configured for the indicated shader stages.  If <stages> is not the
    special value ALL_SHADER_BITS_EXT and has a bit set that is not 
    recognized, the error INVALID_VALUE is generated. If the program object 
    named by <program> was linked without the PROGRAM_SEPARABLE_EXT parameter
    set or was not linked successfully, the error INVALID_OPERATION is 
    generated and the corresponding shader stages in the <pipeline> 
    program pipeline object are not modified.

    If <pipeline> is a name that has been generated (without subsequent
    deletion) by GenProgramPipelinesEXT, but refers to a program pipeline
    object that has not been previously bound, the GL first creates a
    new state vector in the same manner as when BindProgramPipelineEXT
    creates a new program pipeline object. If <pipeline> is not a name
    returned from a previous call to GenProgramPipelinesEXT or if such a
    name has since been deleted by DeleteProgramPipelinesEXT, an 
    INVALID_OPERATION error is generated.

    The command

        void ActiveShaderProgramEXT(uint pipeline, uint program);

    sets the linked program named by <program> to be the active program
    (discussed later in the secion 2.10.4) for the program pipeline
    object <pipeline>.  If <program> has not been successfully linked,
    the error INVALID_OPERATION is generated and active program is not
    modified.

    If <pipeline> is a name that has been generated (without subsequent
    deletion) by GenProgramPipelinesEXT, but refers to a program pipeline
    object that has not been previously bound, the GL first creates a
    new state vector in the same manner as when BindProgramPipelineEXT
    creates a new program pipeline object. If <pipeline> is not a name
    returned from a previous call to GenProgramPipelinesEXT or if such a
    name has since been deleted by DeleteProgramPipelinesEXT, an 
    INVALID_OPERATION error is generated.


    Shader Interface Matching

    When linking a non-separable program object with multiple shader types,
    the outputs of one stage form an interface with the inputs of the next
    stage. These inputs and outputs must typically match in name, type,
    and qualification.  When both sides of an interface are contained in
    the same program object, LinkProgram will detect mismatches on an
    interface and generate link errors.

    With separable program objects, interfaces between shader stages may
    involve the outputs from one program object and the inputs from a
    second program object. For such interfaces, it is not possible to
    detect mismatches at link time, because the programs are linked
    separately. When each such program is linked, all inputs or outputs
    interfacing with another program stage are treated as active. The
    linker will generate an executable that assumes the presence of a
    compatible program on the other side of the interface. If a mismatch
    between programs occurs, no GL error will be generated, but some or all
    of the inputs on the interface will be undefined.

    At an interface between program objects, the inputs and outputs are
    considered to match exactly if and only if:

      * For every user-declared input variable declared, there is an output
        variable declared in the previous shader matching exactly in name,
        type, and qualification.

      * There are no user-defined output variables declared without a matching
        input variable declaration.

    When the set of inputs and outputs on an interface between programs
    matches exactly, all inputs are well-defined unless the corresponding
    outputs were not written in the previous shader. However, any mismatch
    between inputs and outputs results in all inputs being undefined except
    for cases noted below. Even if an input has a corresponding output
    that matches exactly, mismatches on other inputs or outputs may
    adversely affect the executable code generated to read or write the
    matching variable.

    The inputs and outputs on an interface between programs need not match
    exactly when input and output location qualifiers (sections 4.3.6.1 and
    4.3.6.2 of the GLSL Specification) are used.  When using location
    qualifiers, any input with an input location qualifier will be
    well-defined as long as the other program writes to an output with the
    same location qualifier, data type, and qualification.  Also, an input
    will be well-defined if the other program writes to an output matching
    the input in everything but data type as long as the output data type  
    has the same basic component type and more components.  The names of 
    variables need not match when matching by location.  For the purposes 
    of interface matching, an input with a location qualifier is considered 
    to match a corresponding output only if that output has an identical 
    location qualifier.

    Built-in inputs or outputs do not affect interface matching.  Any such 
    built-in inputs are well-defined unless they are derived from built-in 
    outputs not written by the previous shader stage.


    Program Pipeline Object State

    The state required to support program pipeline objects consists of
    a single binding name of the current program pipeline object. This
    binding is initially zero indicating no program pipeline object is
    bound.

    The state of each program pipeline object consists of:

    * Three unsigned integers (initially all zero) are  required to hold
      each respective name of the current vertex stage program, current 
      fragment stage program, and active program respectively.
    * A Boolean holding the status of the last validation attempt,
      initially false.
    * An array of type char containing the information log, initially
      empty.
    * An integer holding the length of the information log."

 -- Section 2.10.4 "Shader Variables" subsection "Vertex Attributes"
 
    Change the first sentence of the second full paragraph (p. 34):
    
    "When a program is linked, any active attributes without a binding
    specified either through BindAttribLocation or explicitly set
    within the shader text will be automatically bound to vertex
    attributes by the GL."
    
    Add the following sentence to the end of that same paragraph:
    
    "If an active attribute has a binding explicitly set within the shader text
    and a different binding assigned by BindAttribLocation, the assignment in 
    the shader text is used."

 -- Section 2.10.4 "Shader Variables" subsection "Uniform Variables"

    Replace the paragraph introducing Uniform* (p. 37):

    "To load values into the uniform variables of the active program object, 
    use the commands

        ... 
    
    If a non-zero program object is bound by UseProgram, it is the
    active program object whose uniforms are updated by these commands.
    If no program object is bound using UseProgram, the active program
    object of the current program pipeline object set by 
    ActiveShaderProgramEXT is the active program object. If the current 
    program pipeline object has no active program or there is no current 
    program pipeline object, then there is no active program.
        
    The given values are loaded into the ... "

    Change the last bullet in the "Uniform Variables" subsection (p. 38) to:

    "* if there is no active program in use."

    Add to the end of the "Uniform Variables" subsection (p. 38): 

    To load values into the uniform variables of a program which may not 
    necessarily be bound, use the commands

        void ProgramUniform{1234}{if}EXT(uint program, int location,
                                         T value);
        void ProgramUniform{1234}uiEXT(uint program, int location,
                                       T value);
        void ProgramUniform{1234}{if}vEXT(uint program, int location,
                                          sizei count, const T *value);
        void ProgramUniform{1234}uivEXT(uint program, int location,
                                          sizei count, const uint *value);
        void ProgramUniformMatrix{234}fvEXT(uint program, int location,
                                            sizei count, boolean transpose,
                                            const float *value);
        void ProgramUniformMatrix{2x3,3x2,2x4,
                                  4x2,3x4,4x3}fvEXT(uint program, 
                                                    int location,
                                                    sizei count, 
                                                    boolean transpose,
                                                    const float *value);

    These commands operate identically to the corresponding commands
    above without "Program" in the command name except, rather than
    updating the currently active program object, these "Program"
    commands update the program object named by the initial <program>
    parameter. The remaining parameters following the initial <program>
    parameter match the parameters for the corresponding non-"Program"
    uniform command. If <program> is not the name of a created program
    or shader object, the error INVALID_VALUE is generated. If <program>
    identifies a shader object or a program object that has not been
    linked successfully, the error INVALID_OPERATION is generated.

 -- Section 2.10.5 "Shader Execution" (p. 40) 

    Change the first paragraph:

    "If there is an active program object present for the vertex stage, the 
    executable code for this active program is used to process incoming vertex
    values."

    Change first paragraph of subsection "Validation", p. 41:

    "It is not always possible to determine at link time if a program object 
    can execute successfully, given that LinkProgram can not know the state of 
    the remainder of the pipeline.  Therefore validation is done when the first
    rendering command (DrawArrays or DrawElements) is issued, to determine if 
    the set of active program objects can be executed. If the current set of 
    active program objects cannot be executed, no primitives
    are processed and the error INVALID_OPERATION will be generated."

    Add to the list in the second paragraph of subsection "Validation" (p. 41):

    "* A program object is active for at least one, but not all of the
      shader stages that were present when the program was linked.

    * There is no current unified program object and the current program
      pipeline object includes a program object that was relinked since
      being applied to the pipeline object via UseProgramStagesEXT with the
      PROGRAM_SEPARABLE_EXT parameter set to FALSE."

    Add after the description of ValidateProgram in subsection
    "Validation":

    "Separable program objects may have validation failures that cannot
    be detected without the complete program pipeline. Mismatched
    interfaces, improper usage of program objects together, and the same
    state-dependent failures can result in validation errors for such
    program objects. As a development aid, use the command

        void ValidateProgramPipelineEXT(uint pipeline);
    
    to validate the program pipeline object <pipeline> against the
    current GL state. Each program pipeline object has a boolean status,
    VALIDATE_STATUS, that is modified as a result of validation. This
    status can be queried with GetProgramPipelineivEXT (See section 6.1.8).
    If validation succeeded, the program pipeline object is guaranteed
    to execute given the current GL state. 

    If <pipeline> is a name that has been generated (without subsequent
    deletion) by GenProgramPipelinesEXT, but refers to a program pipeline
    object that has not been previously bound, the GL first creates a
    new state vector in the same manner as when BindProgramPipelineEXT
    creates a new program pipeline object. If <pipeline> is not a name
    returned from a previous call to GenProgramPipelinesEXT or if such a
    name has since been deleted by DeleteProgramPipelinesEXT, an 
    INVALID_OPERATION error is generated.

Additions to Chapter 3 of the OpenGL ES 2.0 Specification (Rasterization)

 -- Section 3.8 "Fragment Shaders" (p. 86) 

    Replace the last paragraph with:

    "When the program object currently in use for the fragment stage
    includes a fragment shader, its fragment shader is considered active and
    is used to process fragments. If the current fragment stage program
    object has no fragment shader or no program object is current for the
    fragment stage, the results of fragment shader execution are undefined."
    
Additions to Chapter 4 of the OpenGL ES 2.0 Specification (Per-Fragment 
Operations and the Frame Buffer)

    None

Additions to Chapter 5 of the OpenGL ES 2.0 Specification (Special Functions)

    None

Additions to Chapter 6 of the OpenGL ES 2.0 Specification (State and State 
Requests)

 -- Section 6.1.8 "Shader and Program Queries"
    
    Add to GetProgramiv description:

    "If <pname> is PROGRAM_SEPARABLE_EXT, TRUE is returned if the program has
    been flagged for use as a separable program object that can be bound
    to individual shader stages with UseProgramStagesEXT."

    Add after GetProgramiv description:

    "The command 

        boolean IsProgramPipelineEXT(uint pipeline);

    returns TRUE if <pipeline> is the name of a program pipeline object.
    If <pipeline> is zero, or a non-zero value that is not the name of a
    program pipeline object, IsProgramPipelineEXT returns FALSE. No error
    is generated if <pipeline> is not a valid program pipeline object
    name.

    The command

        GetProgramPipelineivEXT(uint pipeline, enum pname, int *params);

    returns properties of the program pipeline object named <pipeline>
    in <params>. The parameter value to return is specified by <pname>.

    If <pipeline> is a name that has been generated (without subsequent
    deletion) by GenProgramPipelinesEXT, but refers to a program pipeline
    object that has not been previously bound, the GL first creates a
    new state vector in the same manner as when BindProgramPipelineEXT
    creates a new program pipeline object. If <pipeline> is not a name
    returned from a previous call to GenProgramPipelinesEXT or if such a
    name has since been deleted by DeleteProgramPipelinesEXT, an 
    INVALID_OPERATION error is generated.

    If <pname> is ACTIVE_PROGRAM_EXT, the name of the active program
    object of the program pipeline object is returned.
    
    If <pname> is VERTEX_SHADER, the name of the current program
    object for the vertex shader type of the program pipeline object is
    returned.
       
    If <pname> is FRAGMENT_SHADER, the name of the current program
    object for the fragment shader type of the program pipeline object
    is returned.
       
    If <pname> is VALIDATE_STATUS, the validation status of the
    program pipeline object, as determined by ValidateProgramPipelineEXT
    (see section 2.10.5) is returned.

    If <pname> is INFO_LOG_LENGTH, the length of the info log,
    including a null terminator, is returned. If there is no info log,
    zero is returned."
    
    Change paragraph describing GetShaderInfoLog and GetProgram:

    "A string that contains information about the last compilation
    attempt on a shader object, last link or validation attempt on a
    program object, or last validation attempt on a program pipeline
    object, called the info log, can be obtained with the commands

        void GetShaderInfoLog(uint shader, sizei bufSize,
                              sizei *length, char *infoLog);
        void GetProgramInfoLog(uint program, sizei bufSize,
                              sizei *length, char *infoLog);
        void GetProgramPipelineInfoLogEXT(uint pipeline, sizei bufSize,
                                          sizei *length, char *infoLog);

    These commands return the info log string in <infoLog>. This string
    will be null-terminated. The actual number of characters written
    into <infoLog>, excluding the null terminator, is returned in
    <length>. If <length> is NULL, then no length is returned. The
    maximum number of characters that may be written into <infoLog>,
    including the null terminator, is specified by <bufSize>. The number
    of characters in the info log can be queried with GetShaderiv,
    GetProgramiv, or GetProgramPipelineivEXT with INFO_LOG_LENGTH. If
    <shader> is a shader object, the returned info log will either be an
    empty string or it will contain information about the last compil-
    ation attempt for that object. If <program> is a program object, the
    returned info log will either be an empty string or it will contain
    information about the last link attempt or last validation attempt
    for that object. If <pipeline> is a program pipeline object, the
    returned info log will either be an empty string or it will contain
    information about the last validation attempt for that object.

Additions to Appendix D of the OpenGL ES 2.0 Specification (Shared Objects and 
Multiple Contexts)

    (add sentence to third paragraph, p. 164) Program pipeline objects are not
    shared.

Additions to the OpenGL ES Shading Language Specification, Version 1.0.17

    Including the following line in a shader can be used to control
    the language feature described in thie extension:

        #extension GL_EXT_separate_shader_objects : <behavior>

    where <behavior> is as described in section 3.3.

    A new preprocessor #define is added to the OpenGL ES Shading Language:

        #define GL_EXT_separate_shader_objects 1

 -- Section 4.3.3 "Attribute" (page 30):

    Add new section 4.3.3.1 "Attribute Layout Qualifiers"
    
    "Vertex shaders allow location layout qualifiers on attribute variable
    declarations.  The location layout qualifier identifier for vertex shader 
    attributes is:

      layout-qualifier-id
        location = integer-constant

    Only one argument is accepted.  For example,

      layout(location = 3) attribute vec4 normal;

    establishes that the vertex shader attribute <normal> is copied in from
    vector location number 3.

    If an input variable with no location assigned in the shader text has a
    location specified through the OpenGL ES API, the API-assigned location will
    be used.  Otherwise, such variables will be assigned a location by the
    linker.  See section 2.10.4 of the OpenGL Specification for more details.
    A link error will occur if an attribute variable is declared in multiple
    vertex shaders with conflicting locations.


 -- Section 4.3.5 "Varying" (page 31):

    Add to the end of the section:

    "When an interface between shader stages is formed using shaders from two
    separate program objects, it is not possible to detect mismatches between
    vertex shader varying outputs and fragment shader varying inputs when the 
    programs are linked. When there are mismatches between inputs and outputs 
    on such interfaces, the values passed across the interface will be 
    partially or completely undefined. Shaders can ensure matches across such 
    interfaces either by using varying layout qualifiers (Section 4.3.5.1) or 
    by using identical varying declarations.  Complete rules for interface
    matching are found in the "Shader Interface Matching" portion of section
    2.10.PPO of the OpenGL Specification."

    Add new section 4.3.5.1 "Varying Layout Qualifiers"

    "All shaders allow location layout qualifiers on varying variable
    declarations. The location layout qualifier identifier for varyings is:

        layout-qualifier-id
            location = integer-constant

    Only one argument is accepted. For example,
        layout(location = 3) varying vec4 normal;

    establishes that the shader varying <normal> is assigned to vector
    location number 3.

    If the declared varying is an array of size <n> and each element takes up
    <m> locations, it will be assigned <m>*<n> consecutive locations starting
    with the location specified.  For example,

        layout(location = 6) varying vec4 colors[3];

    will establish that the input <colors> is assigned to vector
    location numbers 6, 7, and 8.

    If the declared varying is an <n>x<m> matrix, it will be assigned multiple 
    locations starting with the location specified. The number of locations 
    assigned for each matrix will be the same as for an <n>-element array of 
    <m>-component vectors. For example,

        layout(location = 9) varying mat4 transforms[2];

    will establish that varying <transforms> is assigned to vector location
    numbers 9-16, with transforms[0] being assigned to locations 9-12 and
    transforms[1] being assigned to locations 13-16.

    The number of varying locations available to a shader is limited to the 
    implementation-dependent advertised maximum varying vector count.
    A program will fail to link if any attached shader uses a location greater
    than or equal to the number of supported locations, unless
    device-dependent optimizations are able to make the program fit within
    available hardware resources.

    A program will fail to link if any two varying variables are assigned to 
    the same location, or if explicit location assignments leave the linker 
    unable to find space for other variables without explicit assignments."

Dependencies on OpenGL ES 3.0

    If OpenGL ES 3.0 isn't present, references to
    ProgramUniform{1234}ui[v] should be ignored. Also, any redundant
    language about explicit shader variable locations set within the
    shader text should also be ignored.

    If neither OpenGL ES 3.0 nor NV_non_square_matrices is present,
    references to ProgramUniformMatrix{2x3,3x2,2x4,4x2,3x4,4x3}fv
    should be ignored.
    
    The behavior of mixing GLSL ES 1.00 shaders with GLSL ES 3.00 shaders
    in the same rendering pipeline is undefined.
    
    If the GL is OpenGL ES 3.0, make the following changes:
    
    Replace this sentence from the first paragraph of section 2.10.PPO Shader 
    Interface Matching:
    
    "These inputs and outputs must typically match in name, type, and 
    qualification."
    
    with the following language:
    
    "An output variable is considered to match an input variable in the
    subequent shader if:

        * the two variables match in name, type, and qualification; or

        * the two variables are declared with the same location layout
          qualifier and match in type and qualification."
    
    Add this paragraph after the first paragraph of section 2.10.PPO:
    
    "Variables declared as structures are considered to match in type if and 
    only if structure members match in name, type, qualification, and 
    declaration order.  Variables declared as arrays are considered to match 
    in type only if both declarations specify the same element type and array 
    size.  The rules for determining if variables match in qualification are
    found in the OpenGL Shading Language Specification."
    
    Replace the last paragraph from section 2.10.PPO:
    
    "Built-in inputs or outputs do not affect interface matching.  Any such 
    built-in inputs are well-defined unless they are derived from built-in 
    outputs not written by the previous shader stage."
    
    with the following new language:
    
    "When using GLSL ES 1.00 shaders, built-in inputs or outputs do not affect 
    interface matching. Any such built-in inputs are well-defined unless they 
    are derived from built-in outputs not written by the previous shader stage.
    
    When using GLSL ES 3.00 shaders in separable programs, gl_Position and 
    gl_PointSize built-in outputs must be redeclared according to Section 7.5 
    of the OpenGL Shading Language Specification. Other built-in inputs or 
    outputs do not affect interface matching. Any such built-in inputs are 
    well-defined unless they are derived from built-in outputs not written by 
    the previous shader stage."
    
    and add to GLSL ES 3.00 new section 7.5, Built-In Redeclaration and 
    Separable Programs:
    
    "The following vertex shader outputs may be redeclared at global scope to
    specify a built-in output interface, with or without special qualifiers:

        gl_Position
        gl_PointSize

      When compiling shaders using either of the above variables, both such
      variables must be redeclared prior to use.  ((Note:  This restriction
      applies only to shaders using version 300 that enable the
      EXT_separate_shader_objects extension; shaders not enabling the
      extension do not have this requirement.))  A separable program object
      will fail to link if any attached shader uses one of the above variables
      without redeclaration."

Dependencies on NV_non_square_matrices

    If NV_non_square_matrices is supported,
    ProgramUniformMatrix{2x3,3x2,2x4,4x2,3x4,4x3}fvEXT is supported in
    OpenGL ES 2.0.

Errors

    UseProgramStagesEXT generates INVALID_OPERATION if the program
    parameter has not been successfully linked.

    UseProgramStagesEXT generates INVALID_VALUE if <stages> has a bit
    set for any other than VERTEX_SHADER_BIT_EXT or 
    FRAGMENT_SHADER_BIT_EXT, unless <stages> is ALL_SHADER_BITS_EXT.

    ActiveShaderProgramEXT generates INVALID_OPERATION if <program>
    has not been successfully linked.

    DrawArrays and DrawElements generate INVALID_OPERATION when a program 
    object with multiple attached shaders is active for one or more, but not 
    all of the shader program types corresponding to the shaders that are
    attached.

New State

    Add to table 6.15 (Program Object State):
    
                                                       Initial
    Get Value           Type  Get Command              Value    Description               Sec   
    ------------------  ----  -----------------------  -------  ------------------------  ------
    PROGRAM_PIPELINE_-  Z+    GetIntegerv              0        Current program pipeline  2.10.PPO
    BINDING_EXT                                                 object binding

    Add new table 6.PPO (Program Pipeline Object State):

                                                            Initial
    Get Value           Type  Get Command                   Value    Description               Sec   
    ------------------  ----  ----------------------------  -------  ------------------------  ------
    ACTIVE_PROGRAM_EXT  Z+    GetProgramPipelineivEXT       0        The program object        2.10.PPO
                                                                     that Uniform* commands
                                                                     update when PPO bound
    VERTEX_SHADER       Z+    GetProgramPipelineivEXT       0        Name of current vertex    2.10.PPO
                                                                     shader program object
    FRAGMENT_SHADER     Z+    GetProgramPipelineivEXT       0        Name of current fragment  2.10.PPO
                                                                     shader program object
    VALIDATE_STATUS     B     GetProgramPipelineivEXT       FALSE    Validate status of        2.10.PPO
                                                                     program pipeline object
                        S     GetProgramPipelineInfoLogEXT  empty    Info log for program      6.1.8
                                                                     pipeline object                                                      
    INFO_LOG_LENGTH     Z+    GetProgramPipelineivEXT       0        Length of info log        6.1.8
                                                    
New Implementation Dependent State

    None

Issues

    See ARB_separate_shader_objects extension specification for issues
    documented while working on the original desktop OpenGL extension.
    
Revision History

    Rev.    Date      Author     Changes
    ----  ----------  ---------  ---------------------------------------------
    1     2011-06-17  Benj       Initial revision for ES based on
                                 ARB_separate_shader_objects extension.

    2     2011-07-22  Benj       Rename APPLE to EXT, specify that PPOs are
                                 not shareable.

    3     2011-07-26  Benj       Add VALIDATE_STATUS to state tables and
                                 GetProgramPipelineivEXT description, remove 
                                 the language erroneously deleting
                                 CURRENT_PROGRAM.

    4     2013-03-07  Jon Leech  Added note about the unrelated OpenGL 
                                 extension of the same name.             

    5     2013-03-25  Benj       Add interactions with OpenGL ES 3.0.
    6     2013-09-20  dkoch      Add interactions with NV_non_square_matrices.
    7     2013-11-08  marka      Clarify ProgramUniform*ui availability.
