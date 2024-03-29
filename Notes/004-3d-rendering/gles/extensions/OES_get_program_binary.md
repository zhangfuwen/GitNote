# OES_get_program_binary

Name

    OES_get_program_binary

Name Strings

    GL_OES_get_program_binary

Contributors

    Acorn Pooley
    Aske Simon Christensen
    David Garcia
    Georg Kolling
    Jason Green
    Jeremy Sandmel
    Joey Blankenship
    Mark Callow
    Robert Simpson
    Tom Olson

Contact

    Benj Lipchak, AMD (benj.lipchak 'at' amd.com)

Notice

    Copyright (c) 2007-2013 The Khronos Group Inc. Copyright terms at
        http://www.khronos.org/registry/speccopyright.html

Specification Update Policy

    Khronos-approved extension specifications are updated in response to
    issues and bugs prioritized by the Khronos OpenGL ES Working Group. For
    extensions which have been promoted to a core Specification, fixes will
    first appear in the latest version of that core Specification, and will
    eventually be backported to the extension document. This policy is
    described in more detail at
        https://www.khronos.org/registry/OpenGL/docs/update_policy.php

Status

    Ratified by the Khronos BOP, May 29, 2008.

Version

    Last Modified Date: June 24, 2020
    Revision: #16

Number

    OpenGL ES Extension #47

Dependencies

    OpenGL ES 2.0 is required.

    Written based on the wording of the OpenGL ES 2.0 specification.

Overview

    This extension introduces two new commands.  GetProgramBinaryOES empowers an
    application to use the GL itself as an offline compiler.  The resulting
    program binary can be reloaded into the GL via ProgramBinaryOES.  This is a
    very useful path for applications that wish to remain portable by shipping
    pure GLSL source shaders, yet would like to avoid the cost of compiling
    their shaders at runtime.  Instead an application can supply its GLSL source
    shaders during first application run, or even during installation.  The
    application then compiles and links its shaders and reads back the program
    binaries.  On subsequent runs, only the program binaries need be supplied!
    Though the level of optimization may not be identical -- the offline shader
    compiler may have the luxury of more aggressive optimization at its
    disposal -- program binaries generated online by the GL are interchangeable
    with those generated offline by an SDK tool.

    Note that an implementation supporting this extension need not include an
    online compiler.  That is, it is not required to support loading GLSL shader
    sources via the ShaderSource command.  A query of boolean value
    SHADER_COMPILER can be used to determine if an implementation supports a
    shader compiler.  If not, the GetProgramBinaryOES command is rendered
    virtually useless, but the ProgramBinaryOES command may still be used by
    vendor extensions as a standard method for loading offline-compiled program
    binaries.


Issues

    1. Why introduce a new entrypoint for loading binaries when ShaderBinary
       is already part of the core spec and permits loading binary shader pairs?

    RESOLVED: There are several reasons:
      - Shader objects are taken out of the equation, since they're not
        relevant to wholesale program object replacement.
      - Implicit links during retrieval are no longer needed since we don't
        need to keep shader object state in sync with program object state.
      - Explicit links during program object reload are no longer needed since
        the program binary is pre-linked and ready to run.
      - The number of API calls needed to load program objects is much fewer.
      - Complex error detection needed by the previous proposal is eliminated.
      - No change to the retrieval/reload path is needed when new shader stages
        are introduced by future extensions.
      - This is a more elegant mapping for what we're trying to achieve!

    2. Do we need to consider state dependencies when using this extension?

    RESOLVED: No more than you do when using GLSL source shaders.  A program
    binary retrieved with GetProgramBinaryOES can be expected to work regardless
    of the current GL state in effect at the time it was retrieved with
    GetProgramBinaryOES, loaded with ProgramBinaryOES, installed as part of
    render state with UseProgram, or used for drawing with DrawArrays or
    DrawElements.

    However, some implementations have internal state dependencies that affect
    both GLSL source shaders and program binaries, causing them to run out of
    resources when confronted by combinations of certain GL state and certain
    shader program characteristics.  An application need be concerned no more
    with these issues when using program binaries than when using GLSL source
    shaders.

    3. How are shader objects involved, if at all?

    RESOLVED: Any shader objects attached to the program object at the time
    GetProgramBinaryOES or ProgramBinaryOES is called are ignored.  (See also
    Issue 4.)

    The program binary retrieved by GetProgramBinaryOES is the one installed
    during the most recent call to LinkProgram or ProgramBinaryOES, i.e. the one
    which would go into effect if we were to call UseProgram.  Attaching
    different shader objects after the most recent call to LinkProgram is
    inconsequential.

    4. Should we throw an error as a programming aid if there are shader objects
       attached to the program object when ProgramBinaryOES is called?

    RESOLVED: No, they are irrelevant but harmless, and GL precedent is to throw
    errors on bad state combinations, not on harmless ones.  Besides, the
    programmer should discover pretty quickly that they're getting the wrong
    shader, if they accidentally called ProgramBinaryOES instead of LinkProgram.
    Also, an app may intentionally leave the attachments in place if it for some
    reason is switching back and forth between loading a program object with
    program binaries, and loading it with compiled GLSL shaders.

    5. Where are the binary formats defined and described?

    RESOLVED: This extension provides a common infrastructure for retrieving and
    loading program binaries.  A vendor extension must also be present in order
    to define one or more binary formats, thereby populating the list of
    PROGRAM_BINARY_FORMATS_OES.  The <binaryFormat> returned by
    GetProgramBinaryOES is always one of the binary formats in this list.  If
    ProgramBinaryOES is called with a <binaryFormat> not in this list, the
    implementation will throw an INVALID_ENUM error.

    The beauty of this extension, however, is that an application does not need
    to be aware of the vendor extension on any given implementation.  It only
    needs to retrieve a program binary with an anonymous <binaryFormat> and
    resupply that same <binaryFormat> when loading the program binary.

    6. Under what conditions might a call to ProgramBinaryOES fail?

    RESOLVED: Even if a program binary is successfully retrieved with
    GetProgramBinaryOES and then in a future run the program binary is
    resupplied with ProgramBinaryOES, and all of the parameters are correct,
    the program binary load may still fail.

    This can happen if there has been a change to the hardware or software on
    the system, such as a hardware upgrade or driver update.  In this case the
    PROGRAM_BINARY_FORMATS_OES list may no longer contain the binary format
    associated with the cached program binary, and INVALID_ENUM will be thrown
    if the cached program binary format is passed into ProgramBinaryOES anyway.

    Even if the cached program binary format is still valid, ProgramBinaryOES
    may still fail to load the cached binary.  This is the driver's way of
    signaling to the app that it needs to recompile and recache its program
    binaries because there has been some important change to the online
    compiler, such as a bug fix or a significant new optimization.

    7. Can BindAttribLocation be called after ProgramBinaryOES to remap an
       attribute location used by the program binary?

    RESOLVED: No.  BindAttribLocation only affects the result of a subsequent
    call to LinkProgram.  LinkProgram operates on the attached shader objects
    and replaces any program binary loaded prior to LinkProgram.  So there is no
    mechanism to remap an attribute location after loading a program binary.

    However, an application is free to remap an attribute location prior to
    retrieving the program binary.  By calling BindAttribLocation followed by
    LinkProgram, an application can remap the attribute location.  If this is
    followed by a call to GetProgramBinaryOES, the retrieved program binary will
    include the desired attribute location assignment.

New Procedures and Functions

    void GetProgramBinaryOES(uint program, sizei bufSize, sizei *length,
                             enum *binaryFormat, void *binary);

    void ProgramBinaryOES(uint program, enum binaryFormat,
                          const void *binary, int length);

New Tokens

    Accepted by the <pname> parameter of GetProgramiv:

        PROGRAM_BINARY_LENGTH_OES                   0x8741

    Accepted by the <pname> parameter of GetBooleanv, GetIntegerv, and
    GetFloatv:

        NUM_PROGRAM_BINARY_FORMATS_OES              0x87FE
        PROGRAM_BINARY_FORMATS_OES                  0x87FF

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Update section 2.15, replace first sentence of last paragraph with:

    "OpenGL ES 2.0 provides interfaces to directly load pre-compiled shader
    binaries, to directly load pre-linked program binaries, or to load the
    shader sources and compile them."

    Add section 2.15.4, Program Binaries

    "The command

        void GetProgramBinaryOES(uint program, sizei bufSize, sizei *length,
                                 enum *binaryFormat, void *binary);

    returns the program object's executable, henceforth referred to as its
    program binary.  The maximum number of bytes that may be written into
    <binary> is specified by <bufSize>.  If <bufSize> is less than the number of
    bytes in the program binary, then 0 is returned in <length>, and an
    INVALID_OPERATION error is thrown.  Otherwise, the actual number of bytes
    written into <binary> is returned in <length> and its format is returned in
    <binaryFormat>.  If <length> is NULL, then no length is returned.

    The number of bytes in the program binary can be queried by calling
    GetProgramiv with <pname> PROGRAM_BINARY_LENGTH_OES.  When a program
    object's LINK_STATUS is FALSE, its program binary length is zero, and a call
    to GetProgramBinaryOES will generate an INVALID_OPERATION error.

    The command

        void ProgramBinaryOES(uint program, enum binaryFormat,
                              const void *binary, int length);

    loads a program object with a program binary previously returned from
    GetProgramBinaryOES.  This is useful for future instantiations of the GL to
    avoid online compilation, while still using OpenGL Shading Language source
    shaders as a portable initial format.  <binaryFormat> and <binary> must be
    those returned by a previous call to GetProgramBinaryOES, and <length> must
    be the length of the program binary as returned by GetProgramBinaryOES or
    GetProgramiv with <pname> PROGRAM_BINARY_LENGTH_OES.  The program binary
    will fail to load if these conditions are not met.

    An implementation may reject a program binary if it determines the program
    binary was produced by an incompatible or outdated version of the compiler.
    In this case the application should fall back to providing the original
    OpenGL Shading Language source shaders, and perhaps again retrieve the
    program binary for future use.

    A program object's program binary is replaced by calls to LinkProgram or
    ProgramBinaryOES.  Either command sets the program object's LINK_STATUS to
    TRUE or FALSE, as queried with GetProgramiv, to reflect success or failure.
    Either command also updates its information log, queried with
    GetProgramInfoLog, to provide details about warnings or errors.

    If ProgramBinaryOES failed, any information about a previous link or load of
    that program object is lost.  Thus, a failed load does not restore the old
    state of <program>.

    Note that ProgramBinaryOES disregards any shader objects attached to the
    program object, as these shader objects are used only by LinkProgram.

    Queries of values NUM_PROGRAM_BINARY_FORMATS and PROGRAM_BINARY_FORMATS
    return the number of program binary formats and the list of program binary
    format values supported by an implementation.  The <binaryFormat> returned
    by GetProgramBinaryOES must be present in this list."

GLX Protocol

    None.

Errors

    INVALID_OPERATION error is generated if GetProgramBinaryOES is called when
    the program object, <program>, does not contain a valid program binary as
    reflected by its LINK_STATUS state; if <bufSize> is not big enough to
    contain the entire program binary; or if the value of
    NUM_PROGRAM_BINARY_FORMATS is zero.

New State

    (table 6.25, Program Object State) add the following:

    Get Value                  Type  Get Command   Initial Value  Description               Section
    -------------              ----  -----------   -------------  -----------               -------
    PROGRAM_BINARY_LENGTH_OES  Z+    GetProgramiv  0              Length of program binary  2.15.4

    (table 6.28, Implementation Dependent Values) add the following:

    Get Value                       Type  Get Command  Minimum Value  Description                        Section
    -------------                   ----  -----------  -------------  -----------                        -------
    PROGRAM_BINARY_FORMATS_OES      0+*Z  GetIntegerv  N/A            Enumerated program binary formats  2.15.4
    NUM_PROGRAM_BINARY_FORMATS_OES  Z     GetIntegerv  0              Number of program binary formats   2.15.4

    (table 6.29, Implementation Dependent Values (cont.)) add the following:

    Get Value      Type  Get Command          Minimum Value  Description             Section
    -------------  ----  -----------          -------------  -----------             -------
    Binary format  Z1    GetProgramBinaryOES  N/A            Binary format returned  2.15.2

Sample Usage

    void retrieveProgramBinary(const GLchar* vsSource, const GLchar* fsSource,
                               const char* myBinaryFileName,
                               GLenum* binaryFormat)
    {
        GLuint  newFS, newVS;
        GLuint  newProgram;
        GLchar* sources[1];
        GLint   success;

        //
        //  Create new shader/program objects and attach them together.
        //
        newVS = glCreateShader(GL_VERTEX_SHADER);
        newFS = glCreateShader(GL_FRAGMENT_SHADER);
        newProgram = glCreateProgram();
        glAttachShader(newProgram, newVS);
        glAttachShader(newProgram, newFS);

        //
        //  Supply GLSL source shaders, compile, and link them
        //
        sources[0] = vsSource;
        glShaderSource(newVS, 1, sources, NULL);
        glCompileShader(newVS);

        sources[0] = fsSource;
        glShaderSource(newFS, 1, sources, NULL);
        glCompileShader(newFS);

        glLinkProgram(newProgram);
        glGetProgramiv(newProgram, GL_LINK_STATUS, &success);

        if (success)
        {
            GLint   binaryLength;
            void*   binary;
            FILE*   outfile;

            //
            //  Retrieve the binary from the program object
            //
            glGetProgramiv(newProgram, GL_PROGRAM_BINARY_LENGTH_OES, &binaryLength);
            binary = (void*)malloc(binaryLength);
            glGetProgramBinaryOES(newProgram, binaryLength, NULL, binaryFormat, binary);

            //
            //  Cache the program binary for future runs
            //
            outfile = fopen(myBinaryFileName, "wb");
            fwrite(binary, binaryLength, 1, outfile);
            fclose(outfile);
            free(binary);
        }
        else
        {
            //
            // Fallback to simpler source shaders?  Take my toys and go home?
            //
        }

        //
        // Clean up
        //
        glDeleteShader(newVS);
        glDeleteShader(newFS);
        glDeleteProgram(newProgram);
    }

    void loadProgramBinary(const char* myBinaryFileName, GLenum binaryFormat,
                           GLuint progObj)
    {
        GLint   binaryLength;
        void*   binary;
        GLint   success;
        FILE*   infile;

        //
        //  Read the program binary
        //
        infile = fopen(myBinaryFileName, "rb");
        fseek(infile, 0, SEEK_END);
        binaryLength = (GLint)ftell(infile);
        binary = (void*)malloc(binaryLength);
        fseek(infile, 0, SEEK_SET);
        fread(binary, binaryLength, 1, infile);
        fclose(infile);

        //
        //  Load the binary into the program object -- no need to link!
        //
        glProgramBinaryOES(progObj, binaryFormat, binary, binaryLength);
        free(binary);

        glGetProgramiv(progObj, GL_LINK_STATUS, &success);

        if (!success)
        {
            //
            // Something must have changed since the program binaries
            // were cached away.  Fallback to source shader loading path,
            // and then retrieve and cache new program binaries once again.
            //
        }
    }

Revision History

    #16    24/06/2020    Arthur Tombs    Fix typo: pass binaryLength by value
                                         instead of by pointer in example code
    #15    01/11/2019    Jon Leech       Add an error for ProgramBinary if there
                                         are no binary formats (Bug 16155).
    #14    10/08/2013    Jon Leech       Change GLvoid -> void (Bug 10412).
    #13    06/02/2008    Benj Lipchak    Fix typo: GLint -> int, update status.
    #12    05/07/2008    Benj Lipchak    Add Issue about BindAttribLocation.
    #11    04/03/2008    Benj Lipchak    Fix memory leaks in sample code.
    #10    03/27/2008    Benj Lipchak    Mark spec as ratified by the WG, add
                                         new issues, and update sample code.
    #09    03/13/2008    Benj Lipchak    Many minor updates!  Most notably,
                                         introduce PROGRAM_BINARY_FORMATS_OES
                                         and NUM_PROGRAM_BINARY_FORMATS_OES.
    #08    03/12/2008    Benj Lipchak    Rewrite as {Get}ProgramBinaryOES.  Add
                                         issues section.
    #07    02/27/2008    Benj Lipchak    When <bufSize> is too small, throw
                                         error and return 0 in <length>.  Limit
                                         the allowed reasons for subsequent
                                         binary rejection.  Rename to OES and
                                         GetShaderBinary.  Add the LinkProgram
                                         error condition.
    #06    01/10/2008    Benj Lipchak    Clarify that GetProgramInfoLog may be
                                         called after an implicit link, and
                                         clarify that the returned binary pair
                                         must be loaded with a single call to
                                         ShaderBinary or an error is thrown.
    #05    01/08/2008    Benj Lipchak    Clarify program object state after
                                         GetProgramBinaryEXT, fix example code.
    #04    01/02/2008    Benj Lipchak    Split GetProgramBinary into its own
                                         multi-vendor extension proposal.
    #03    11/26/2007    Benj Lipchak    Add sample usage and define tokens.
    #02    10/22/2007    Benj Lipchak    Add error conditions.
    #01    10/14/2007    Benj Lipchak    First draft.
