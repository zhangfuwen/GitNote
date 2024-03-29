# AMD_program_binary_Z400

Name

    AMD_program_binary_Z400

Name Strings

    GL_AMD_program_binary_Z400

Contributors

    Joey Blankenship

Contact

    Benj Lipchak, AMD (benj.lipchak 'at' amd.com)

Status

    Complete.

Version
    
    Last Modified Date: April 9, 2008
    Revision: #6

Number

    48

Dependencies

    OpenGL ES 2.0 is required.

    OES_get_program_binary is required.

    Written based on the wording of the OpenGL ES 2.0 specification.

Overview

    AMD provides an offline shader compiler as part of its suite of SDK tools
    for AMD's Z400 family of embedded graphics accelerator IP.  This extension
    makes available a program binary format, Z400_BINARY_AMD.

    The offline shader compiler accepts a pair of OpenGL Shading Language 
    (GLSL) source shaders: one vertex shader and one fragment shader.  It
    outputs a compiled, optimized, and pre-linked program binary which can then
    be loaded into a program objects via the ProgramBinaryOES command.

    Applications are recommended to use the OES_get_program_binary extension's
    program binary retrieval mechanism for install-time shader compilation where
    applicable.  That cross-vendor extension provides the performance benefits
    of loading pre-compiled program binaries, while providing the portability of
    deploying GLSL source shaders with the application rather than vendor-
    specific binaries.  The details of this extension are obviated by the use
    of that extension.

New Procedures and Functions

    None.

New Tokens

    Accepted by the <binaryFormat> parameter of ProgramBinaryOES:

        Z400_BINARY_AMD                            0x8740

Additions to Chapter 2 of the OpenGL ES 2.0 Specification (OpenGL Operation)

    Add the following paragraph to the end of section 2.15.4:

    "Z400_BINARY_AMD, returned in the list of PROGRAM_BINARY_FORMATS_OES, is a
    format that may be loaded into a program object via ProgramBinaryOES."

    An implementation may reject a Z400_BINARY_AMD program binary by setting the
    LINK_STATUS to FALSE and updating the program object's info log if it
    determines the binary was produced by an incompatible or outdated version of
    the shader compiler."

GLX Protocol

    None.

Errors

    None.

New State

    None.

Sample Usage

    void loadPrecompiledZ400ProgramBinary(const char* myZ400BinaryFileName,
                                          GLuint progObj)
    {
        GLint   binaryLength;
        GLvoid* binary;
        GLint   success;
        FILE*   infile;

        //
        //  Read the program binary
        //
        infile = fopen(myZ400BinaryFileName, "rb");
        fseek(infile, 0, SEEK_END);
        binaryLength = (GLint)ftell(infile);
        binary = (GLvoid*)malloc(binaryLength);
        fseek(infile, 0, SEEK_SET);
        fread(binary, binaryLength, 1, infile);
        fclose(infile);

        //
        //  Load the binary into the program object -- no need to link!
        //
        glProgramBinaryOES(progObj, GL_Z400_BINARY_AMD, binary, binaryLength);
        free(binary);

        glGetProgramiv(progObj, GL_LINK_STATUS, &success);

        if (!success)
        {
            //
            // Fallback to source shaders or gracefully exit.
            //
        }
    }

Revision History

    #06    04/09/2008    Benj Lipchak    Remove INVALID_OPERATION error in favor
                                         of just LINK_STATUS and info log.
                                         Also improve sample code.
    #05    03/12/2008    Benj Lipchak    Reformulate as program binary.
    #04    01/02/2008    Benj Lipchak    Split GetProgramBinary into its own
                                         multi-vendor extension proposal.
    #03    11/26/2007    Benj Lipchak    Add sample usage and define tokens.
    #02    10/22/2007    Benj Lipchak    Add error conditions.
    #01    10/14/2007    Benj Lipchak    First draft.
