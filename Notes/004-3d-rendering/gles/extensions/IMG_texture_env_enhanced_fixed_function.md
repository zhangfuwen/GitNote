# IMG_texture_env_enhanced_fixed_function

Name

    IMG_texture_env_enhanced_fixed_function

Name Strings

    GL_IMG_texture_env_enhanced_fixed_function

Notice

    Copyright Imagination Technologies Limited, 2005.

Contact

    Graham Connor, Imagination Technologies (graham 'dot' connor 'at'
    imgtec 'dot' com)

Status

    Complete

Version
    
    1.0, 11 May 2009

Number

    OpenGL ES Extension #58

Dependencies

    None

    This extension is written against the OpenGL 1.3 Specification. The intention 
    is that this extension is only exposed, within a OpenGL-ES 1.0 impementation 
    and the reader should be aware of the restrictions of OpenGL-ES 1.0 with respect
    to this OpenGL 1.3 extension.

Overview

    This extension adds new texture environment functions to allow use of 
    of blend modes supported in some early MBX-lite devices, including dot3 functionality. 
    It is superceded by OpenGL-ES 1.1 which includes tex_env_combine.

    New functions may be specified by calling TexEnv with the following tokens: 
    MODULATE_COLOR_IMG, RECIP_ADD_SIGNED_ALPHA_IMG, TEXTURE_ALPHA_MODULATE_IMG, 
    FACTOR_ALPHA_MODULATE_IMG, FRAGMENT_ALPHA_MODULATE_IMG, ADD_BLEND_IMG, DOT3_RGBA.

New Procedures and Functions

    None

New Tokens

    Accepted by the <params> parameter of TexEnvf, TexEnvi, TexEnvfv, and
    TexEnvfi when the <pname> parameter value is GL_TEXTURE_ENV_MODE

        MODULATE_COLOR_IMG                           0x8C04
        RECIP_ADD_SIGNED_ALPHA_IMG                   0x8C05
        TEXTURE_ALPHA_MODULATE_IMG                   0x8C06
        FACTOR_ALPHA_MODULATE_IMG                    0x8C07
        FRAGMENT_ALPHA_MODULATE_IMG                  0x8C08
        ADD_BLEND_IMG                                0x8C09
        DOT3_RGBA_IMG                                0x86AF

Additions to Chapter 2 of the GL Specification (OpenGL Operation)

    None

Additions to Chapter 3 of the GL Specification (Rasterization)

    The description of TEXTURE_ENV_MODE in the first paragraph of
    section 3.8.12 should be modified as follows:

    TEXTURE_ENV_MODE may be set to one of REPLACE, MODULATE, DECAL,
    BLEND, ADD, MODULATE_COLOR_IMG, RECIP_ADD_SIGNED_ALPHA_IMG, 
    TEXTURE_ALPHA_MODULATE_IMG, FACTOR_ALPHA_MODULATE_IMG, 
    FRAGMENT_ALPHA_MODULATE_IMG, ADD_BLEND_IMG, DOT3_RGBA_IMG, or COMBINE;

    Table 3.24 is added as follows:

    Base                MODULATE_COLOR_IMG            RECIP_ADD_SIGNED_ALPHA_IMG
    Internal Format     tex func                      tex func  
    ---------------     ------------------            --------------------------

    ALPHA               Cv = Cf                       Cv = Cf      
                        Av = As                       Av = (1-As) + Af - 0.5

    LUMINANCE           Cv = CfCs                     Cv = Cf
    (or 1)              Av = Af                       Av = Af - 0.5          
        
    LUMINANCE_ALPHA     Cv = CfCs                     Cv = Cf      
    (or 2)              Av = As                       Av = (1-As) + Af - 0.5

    INTENSITY           Cv = CfCs                     Cv = Cf      
                        Av = As                       Av = (1-As) + Af - 0.5     

    RGB                 Cv = CfCs                     Cv = Cf
    (or 3)              Av = Af                       Av = Af - 0.5  

    RGBA                Cv = CfCs                     Cv = Cf
    (or 4)              Av = As                       Av = (1-As) + Af - 0.5     


    Base                TEXTURE_ALPHA_MODULATE_IMG    FACTOR_ALPHA_MODULATE_IMG
    Internal Format     tex func                      tex func  
    ---------------     --------------------------    -------------------------

    ALPHA               Cv = ZERO                     Cv = ZERO  
                        Av = As                       Av = Ac

    LUMINANCE           Cv = Cs                       Cv = AcCs
    (or 1)              Av = ONE                      Av = Ac           
        
    LUMINANCE_ALPHA     Cv = AsCs                     Cv = AcCs  
    (or 2)              Av = As                       Av = Ac

    INTENSITY           Cv = AsCs                     Cv = AcCs      
                        Av = As                       Av = Ac

    RGB                 Cv = Cs                       Cv = AcCs
    (or 3)              Av = ONE                      Av = Ac

    RGBA                Cv = AsCs                     Cv = AcCs
    (or 4)              Av = As                       Av = Ac


    Base                FRAGMENT_ALPHA_MODULATE_IMG    ADD_BLEND_IMG
    Internal Format     tex func                       tex func  
    ---------------     -----------------------        -------------

    ALPHA               Cv = ZERO                      Cv = Cf
                        Av = Af                        Av = AfAs

    LUMINANCE           Cv = AfCs                      Cv = Cf + (1 - Af)Cs
    (or 1)              Av = Af                        Av = Af
        
    LUMINANCE_ALPHA     Cv = AfCs                      Cv = Cf + (1 - Af)Cs
    (or 2)              Av = Af                        Av = AfAs

    INTENSITY           Cv = AfCs                      Cv = Cf + (1 - Af)Cs
                        Av = Af                        Av = AfAs

    RGB                 Cv = AfCs                      Cv = Cf + (1 - Af)Cs
    (or 3)              Av = Af                        Av = Af

    RGBA                Cv = AfCs                      Cv = Cf + (1 - Af)Cs
    (or 4)              Av = Af                        Av = AfAs


    Base                DOT3_RGBA_IMG
    Internal Format     tex func    
    ---------------     ---------    

    ALPHA               Undefined    
                        Undefined    

    LUMINANCE           Undefined    
     (or 1)             Undefined    
        
    LUMINANCE_ALPHA     Undefined    
     (or 2)             Undefined    

    INTENSITY           Undefined    
                        Undefined    

    RGB                 Cv = Dot3(Cf,Cs)
    (or 3)              Av = Dot3(Cf,Cs)

    RGBA                Cv = Dot3(Cf,Cs)
    (or 4)              Av = Dot3(Cf, Cs)


    where Dot3(Cf,Cs) evaluates to:

          4((Cfr - 0.5)*(Csr - 0.5) +
            (Cfg - 0.5)*(Csg - 0.5) +
            (Cfb - 0.5)*(Csb - 0.5))


    Table 3.24: Extended Fixed Function Texturing Modes


Additions to Chapter 4 of the GL Specification (Per-Fragment Operations
and the Framebuffer)

    None

Additions to Chapter 5 of the GL Specification (Special Functions)

    None

Additions to Chapter 6 of the GL Specification (State and State Requests)

    The Type of TEXTURE_ENV_MODE in Table 6.17 should be changed to

    2* x Z13

Additions to the GLX / WGL / AGL Specifications

    None

GLX Protocol

    None

Errors

    None

New State

    The Type of TEXTURE_ENV_MODE in Table 6.17 should be changed to

    2* x Z13

New Implementation Dependent State

    None

Revision History

    0.1, 18/12/2003  gdc: First draft.
    0.2, 13/01/2004  gdc: Formatting changes.
    0.3, 25/01/2005  nt:  Updated copyright date.
    1.0, 11/05/2009  bcb: Final tidy up for publish.
  
