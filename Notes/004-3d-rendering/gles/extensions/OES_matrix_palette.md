# OES_matrix_palette

Name

    OES_matrix_palette

Name Strings

    GL_OES_matrix_palette

Contact

    Aaftab Munshi (amunshi@ati.com)

Notice

    Copyright (c) 2004-2013 The Khronos Group Inc. Copyright terms at
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

    Ratified by the Khronos BOP, Aug 5, 2004.

Version

    Version 1, August 2004

Number

    OpenGL ES Extension #12

Dependencies

    OpenGL ES 1.0 is required.

Overview

    This extension adds the ability to support vertex skinning in OpenGL ES.
    A simplified version of the ARB_matrix_palette extension is used to
    define OES_matrix_palette extension.

    This extension allow OpenGL ES to support a palette of matrices.  The matrix
    palette defines a set of matrices that can be used to transform a vertex.
    The matrix palette is not part of the model view matrix stack and is enabled
    by setting the MATRIX_MODE to MATRIX_PALETTE_OES.

    The n vertex units use a palette of m modelview matrices (where n and m are
    constrained to implementation defined maxima.)  Each vertex has a set of n
    indices into the palette, and a corresponding set of n weights.
    Matrix indices and weights can be changed for each vertex.  
    
    When this extension is utilized, the enabled units transform each
    vertex by the modelview matrices specified by the vertices'
    respective indices.  These results are subsequently scaled by the
    weights of the respective units and then summed to create the
    eyespace vertex.   
    
    A similar procedure is followed for normals.  Normals, however,
    are transformed by the inverse transpose of the modelview matrix.
    
IP Status

    Unknown, but believed to be none. 

Issues

    Should this extension be an optional or mandatory extension

        Will be an optional extension since ARB_matrix_palette didn't
        see much usage in OpenGL.

    Should we allow the ability to load the current model view matrix 
    into the matrix palette

        Yes.  This will be very helpful since it makes it very easy
        to load an object heirarchy.  This will also be helpful for JSR184

    Should the Matrix palette be loaded with a new LoadMatrixPalette
    command?

        No, although this provides an easy way to support arbitrary
        palette sizes, the method loses the current (MultMatrix,
        Rotate, Translate, Scale..) matrix functionality.

        Matrices will be Loaded into the palette with current
        functions when MATRIX_MODE is MATRIX_PALETTE_OES.  The current
        palette index is set by an explicit command:
        CurrentPaletteMatrixARB(). 
    
        
    Should the Matrix Palette have a stack?

        Not required, this wastes a lot of space.
        
        
    Should the matrix palette be gettable?

        No.
       
    Should MatrixIndexARB be changed to imply LoadMatrix calls to the
    applicable MODELVIEW_MATRIXn stacks?

        No, the MODELVIEW_MATRIXn matrices are unused when
        MATRIX_PALETTE is enabled.
       
    
    Should there be a way to specify that the modelview matrices
    for two different vertex units are identical?

        Not explicitly, but indexing the matrix palette provides this
        functionality. (Both units will have the same matrix index.)
        
        
New Procedures and Functions

    void CurrentPaletteMatrixOES(uint index)

    void LoadPaletteFromModelViewMatrixOES()
    
    void MatrixIndexPointerOES(int size, enum type, sizei stride, void *pointer)

    void WeightPointerOES(int size, enum type, sizei stride, void *pointer);
 
New Tokens

    Accepted by the <mode> parameter of MatrixMode, and by the
    <cap> parameters of Enable and Disable:

      MATRIX_PALETTE_OES                    0x8840

    Accepted by the <pname> parameters of GetIntegerv:

      MAX_PALETTE_MATRICES_OES              0x8842
      MAX_VERTEX_UNITS_OES                  0x86A4
      CURRENT_PALETTE_MATRIX_OES            0x8843

    The default values for MAX_PALETTE_MATRICES_OES and MAX_VERTEX_UNITS_OES
    are 9 and 3 resp.
      
    Accepted by the <cap> parameters of EnableClientState and DisableClientState and
    by the <pname> parameter of IsEnabled:

      MATRIX_INDEX_ARRAY_OES                0x8844
      WEIGHT_ARRAY_OES                      0x86AD

    Accepted by the <pname> parameter of GetIntegerv:

      MATRIX_INDEX_ARRAY_SIZE_OES           0x8846
      MATRIX_INDEX_ARRAY_TYPE_OES           0x8847
      MATRIX_INDEX_ARRAY_STRIDE_OES         0x8848
      MATRIX_INDEX_ARRAY_BUFFER_BINDING_OES 0x8B9E

      WEIGHT_ARRAY_SIZE_OES                 0x86AB
      WEIGHT_ARRAY_TYPE_OES                 0x86A9
      WEIGHT_ARRAY_STRIDE_OES               0x86AA
      WEIGHT_ARRAY_BUFFER_BINDING_OES       0x889E

    Accepted by the <pname> parameter of GetPointerv:

      MATRIX_INDEX_ARRAY_POINTER_OES        0x8849
      WEIGHT_ARRAY_POINTER_OES              0x86AC

Additions to Chapter 2 of the OpenGL ES 1.0 Specification

    - Added to section 2.8 

          void WeightPointerOES(int size, enum type, sizei stride, void *pointer);

          void MatrixIndexPointerOES(int size, enum type, sizei stride, void *pointer);

        WeightPointerOES & MatrixIndexPointerOES are used to describe the weights and
        matrix indices used to blend corresponding matrices for a given vertex.

        For implementations supporting matrix palette, note that <size> values for
        WeightPointerOES & MatrixIndexPointerOES must be less than or equal to the
        implementation defined value MAX_VERTEX_UNITS_OES.

    - Added to table in section 2.8

        Command                 Sizes                       Types
        -------                 -----                       -----
        WeightPointerOES        1..MAX_VERTEX_UNITS_OES     fixed, float
        MatrixIndexPointerOES   1..MAX_VERTEX_UNITS_OES     ubyte
         

    - (section 2.8) Extend the cap flags passed to EnableClientState/DisableClientState
       to include

          MATRIX_INDEX_ARRAY_OES, or WEIGHT_ARRAY_OES

    - (section 2.10) Add the following:

          "The vertex coordinates that are presented to the GL are termed
           object coordinates. The model-view matrix is applied to these
           coordinates to yield eye coordinates. In implementations with
           matrix palette, the matrices specified by the indices per vertex
           are applied to these coordinates and the weighted sum of the
           results are the eye coordinates. Then another matrix, called the
           projection matrix, is applied to eye coordinates to yield clip
           coordinates.  A perspective division is carried out on clip
           coordinates to yield normalized device coordinates.

           A final viewport transformation is applied to convert these
           coordinates into window coordinates."
    
          "... the vertex's eye coordinates are found as:

            (xe)    n-1               (xo)
            (ye)  =  SUM  w_i * M_i * (yo)
            (ze)    i=0               (zo)
            (we)                      (wo)

          where M_i is the palette matrix associated with the i'th
          Vertex unit:
          
            M_i = MatrixPalette[MatrixIndex[i]],
                     if MATRIX_PALETTE_OES is enabled, and
            M_i = MODELVIEW_MATRIX, otherwise.
            
          w_i is the Vertex's associated weight for vertex unit i:

            w_i = weight_i, if MATRIX_PALETTE_OES is enabled,
                         1, if MATRIX_PALETTE_OES is disabled,

          and,
          
            n = <size> value passed into glMatrixIndexPointerOES."
          

          "The projection matrix and model-view matrices are set
          with a variety of commands. The affected matrix is
          determined by the current matrix mode. The current
          matrix mode is set with

            void MatrixMode( enum mode );
 
          which takes one of the pre-defined constants TEXTURE,
          MODELVIEW, PROJECTION, MATRIX_PALETTE_OES.
 

          In implementations supporting OES_matrix_palette,

             void CurrentPaletteMatrixOES(uint index);
              
          defines which of the palette's matrices is affected by
          subsequent matrix operations when the current matrix mode is
          MATRIX_PALETTE_OES. CurrentPaletteMatrixOES generates the
          error INVALID_VALUE if the <index> parameter is not between
          0 and MAX_PALETTE_MATRICES_OES - 1.

          In implementations supporting OES_matrix_palette,

             void LoadPaletteFromModelViewMatrixOES();

          copies the current model view matrix to a matrix in the matrix
          palette, specified by CurrentPaletteMatrixOES.

          DrawArrays and DrawElements will not render the primitive if
          the matrix palette was enabled and the weights and/or matrix
          index vertex pointers are disabled or are not valid.

          "The state required to implement transformations consists of a
          four-valued integer indicating the current matrix mode, a
          stack of at least two 4 x 4 matrices for each of PROJECTION,
          and TEXTURE with associated stack pointers, a stack of at least
          32 4 x 4 matrices with an associated stack pointer for MODELVIEW,
          and a set of MAX_PALETTE_MATRICES_OES matrices of at least 9
          4 x 4 matrices each for the matrix palette.
          
          Initially, there is only one matrix on each stack, and all
          matrices are set to the identity.  The initial matrix mode
          is MODELVIEW. 

          "When matrix palette is enabled, the normal is transformed
          to eye space by:

                                              n-1
              (nx' ny' nz') = (nx ny nz) Inv ( SUM w_i * Mu_i)
                                              i=0
          
            Alternatively implementations may choose to transform the
          normal to eye-space by:
          
                              n-1
              (nx' ny' nz') =  SUM w_i * (nx ny nz) Inv(Mu_i)
                              i=0

          where Mu_i is the upper leftmost 3x3 matrix taken from the
          modelview for vertex unit i (M_i),
         
               M_i = MatrixPalette[MatrixIndex[i]], 
                         if MATRIX_PALETTE_OES is enabled, and
               M_i = MODELVIEW_MATRIX, otherwise
         
          otherwise.

          weight_i is the vertex's associated weight for vertex unit i,

              w_i = weight_i
                        
          and

              n = <size> value passed into glMatrixIndexPointerOES."


Errors
      
      INVALID_VALUE is generated if the <size> parameter for
      MatrixIndexPointerOES or WeightPointerOES is greater
      than MAX_VERTEX_UNITS_OES.

      INVALID_VALUE is generated if the <count> parameter to
      CurrentPaletteMatrixOES is greater than MAX_PALETTE_MATRICES_OES - 1


New State

(table 6.6, p. 232)

                                       Get          Initial
Get Value                      Type    Command      Value   Description                    
---------                      ----    -------      ------- -----------                   
MATRIX_INDEX_ARRAY_OES         B       IsEnabled    False   matrix index array enable
MATRIX_INDEX_ARRAY_SIZE_OES    Z+      GetIntegerv  0       matrix indices per vertex
MATRIX_INDEX_ARRAY_TYPE_OES    Z+      GetIntegerv  UBYTE   type of matrix index data
MATRIX_INDEX_ARRAY_STRIDE_OES  Z+      GetIntegerv  0       stride between
                                                            matrix indices
MATRIX_INDEX_ARRAY_POINTER_OES Y       GetPointerv  0       pointer to matrix
                                                            index array

WEIGHT_ARRAY_OES               B       IsEnabled    False   weight array enable
WEIGHT_ARRAY_SIZE_OES          Z+      GetIntegerv  0       weights per vertex
WEIGHT_ARRAY_TYPE_OES          Z2      GetIntegerv  FLOAT   type of weight data
WEIGHT_ARRAY_STRIDE_OES        Z+      GetIntegerv  0       stride between weights
                                                            per vertex
WEIGHT_ARRAY_POINTER_OES       Y       GetPointerv  0       pointer to weight array


(table 6.7, p. 233)

                                             Get         Initial
Get Value                              Type  Command      Value  Description 
---------                              ----  -------      -----  -----------

MATRIX_INDEX_ARRAY_BUFFER_BINDING_OES  Z+    GetIntegerv  0      matrix index array
                                                                 buffer binding

WEIGHT_ARRAY_BUFFER_BINDING_OES        Z+    GetIntegerv  0      weight array
                                                                 buffer binding

(table 6.9, p. 235)

                                  Get          Initial
Get Value                   Type  Command      Value    Description 
---------                   ----  -------      -------  -----------

MATRIX_PALETTE_OES          B     IsEnabled    False    matrix palette enable
MAX_PALETTE_MATRICES_OES    Z+    GetIntegerv  9        size of matrix palette
MAX_VERTEX_UNITS_OES        Z+    GetIntegerv  3        number of matrices per vertex
CURRENT_PALETTE_MATRIX_OES  Z+    GetIntegerv  0        transform  index of current 
                                                        modelview matrix in the palette,
                                                        as set by CurrentPaletteMatrixOES()


Revision History



Addendum: Using this extension.

    /* position viewer */
    glMatrixMode(GL_MATRIX_PALETTE_OES);
    glCurrentPaletteMatrixOES(0);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -7.0f);
    glRotatef(yrot, 0.0f, 1.0f, 0.0f);

    glCurrentPaletteMatrixOES(1);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -7.0f);

    glRotatef(yrot, 0.0f, 1.0f, 0.0f);
    glRotatef(zrot, 0.0f, 0.0f, 1.0f);

    glEnable(GL_MATRIX_PALETTE_OES);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    glEnableClientState(GL_MATRIX_INDEX_ARRAY_OES);
    glEnableClientState(GL_WEIGHT_ARRAY_OES);

    glVertexPointer(3, GL_FLOAT, 7 * sizeof(GLfloat), vertexdata);
    glTexCoordPointer(2, GL_FLOAT, 7 * sizeof(GLfloat), vertexdata + 3);
    glWeightPointerOES(2, GL_FLOAT, 7 * sizeof(GLfloat),vertexdata + 5);
    glMatrixIndexPointerOES(2, GL_UNSIGNED_BYTE, 0, matrixindexdata);
        
    for(int i = 0; i < (numSegments << 2) + 2; i ++)
        glDrawArrays(GL_TRIANGLE_FAN, i << 2, 4);

