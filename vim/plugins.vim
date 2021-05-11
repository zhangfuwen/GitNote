" 1. Plugins {{{
"
" 1.1 Installation {{{
" sudo apt install ripgrep  # install rg for global text searhch in Leaderf rg
" sudo apt install cppman; cppman -s cppreference.com; cppman -r
" pip3 install --user neovim
" }}} end Installation
"
call plug#begin('~/.vim/plugged')

" 1.2 Appearance {{{
"Plug 'itchyny/lightline.vim'
Plug 'vim-airline/vim-airline'
Plug 'vim-airline/vim-airline-themes'
let laststatus=2 "永远显示状态栏
let g:airline_powerline_fonts = 1 " 支持 powerline 字体
let g:airline#extensions#tabline#enabled = 1 "显示窗口tab和buffer
let g:airline#extensions#tabline#buffer_nr_show = 1
" let g:airline_theme="badcat"
"AirlineTheme badcat
Plug 'NLKNguyen/papercolor-theme'

Plug 'flazz/vim-colorschemes'
Plug 'ryanoasis/vim-devicons'

Plug 'itchyny/vim-cursorword'

Plug 'octol/vim-cpp-enhanced-highlight'

" 显示缩进层级的竖线
Plug 'Yggdroot/indentLine'
" }}}

" 1.3 Panels {{{
Plug 'scrooloose/nerdtree'
Plug 'majutsushi/tagbar'

" echodoc 状态栏显示文档 {{{
"Plug 'Shougo/echodoc.vim'
set noshowmode
" Or, you could use vim's popup window feature.
"let g:echodoc#enable_at_startup = 1
"let g:echodoc#type = 'popup'
" To use a custom highlight for the popup window,
" change Pmenu to your highlight group
" highlight link EchoDocPopup Pmenu
" When you accept a completion for a function with <c-y>,
" echodoc will display the function signature in the command line
" and highlight the argument position your cursor is in.
" }}}
"
"}}}

" 1.4 PopupTools {{{
Plug 'Yggdroot/LeaderF',{'do':'./install.sh'}
let g:Lf_WindowPosition = 'popup'
Plug 'tiagofumo/vim-nerdtree-syntax-highlight'
Plug 'skywind3000/vim-quickui'
"Plug 'skywind3000/vim-cppman'
Plug 'skywind3000/vim-preview'
"}}}

" 1.5 VCS {{{
Plug 'airblade/vim-gitgutter'
"Plug 'motemen/git-vim'
Plug 'tpope/vim-fugitive'
Plug 'will133/vim-dirdiff'
Plug 'gregsexton/gitv'
" }}}

" 1.6 Text objects {{{
" 它新定义的文本对象主要有：
"   i, 和 a, ：参数对象，写代码一半在修改，现在可以用 di, 或 ci, 一次性删除/改写当前参数
"   ii 和 ai ：缩进对象，同一个缩进层次的代码，可以用 vii 选中，dii / cii 删除或改写
"   if 和 af ：函数对象，可以用 vif / dif / cif 来选中/删除/改写函数的内容
Plug 'kana/vim-textobj-user'
Plug 'kana/vim-textobj-indent'
Plug 'kana/vim-textobj-syntax'
Plug 'kana/vim-textobj-function', { 'for':['c', 'cpp', 'vim', 'java'] }
Plug 'sgur/vim-textobj-parameter'
" }}}

" 1.7 Async lint engine {{{
Plug 'w0rp/ale'
let g:ale_linters_explicit = 1
let g:ale_completion_delay = 500
let g:ale_echo_delay = 20
let g:ale_lint_delay = 500
let g:ale_echo_msg_format = '[%linter%] %code: %%s'
let g:ale_lint_on_text_changed = 'normal'
let g:ale_lint_on_insert_leave = 1
let g:airline#extensions#ale#enabled = 1

let g:ale_c_gcc_options = '-Wall -O2 -std=c99'
let g:ale_cpp_gcc_options = '-Wall -O2 -std=c++14'
let g:ale_c_cppcheck_options = ''
let g:ale_cpp_cppcheck_options = ''

let g:ale_sign_error = "\ue009\ue009"
hi! clear SpellBad
hi! clear SpellCap
hi! clear SpellRare
hi! SpellBad gui=undercurl guisp=red
hi! SpellCap gui=undercurl guisp=blue
hi! SpellRare gui=undercurl guisp=magenta
" }}}

" 1.8 Auto completion {{{
if(has('python3'))
" completion {{{
"    if has('nvim')
"        Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
"    else
"        Plug 'Shougo/deoplete.nvim'
"        Plug 'roxma/nvim-yarp'
"        Plug 'roxma/vim-hug-neovim-rpc'
"    endif
"    let g:deoplete#enable_at_startup = 1
"    Plug 'Shougo/deoplete-clangx'
"    Plug 'deoplete-plugins/deoplete-zsh'
"    Plug 'fszymanski/deoplete-emoji'
    " }}}

" coc conquer of completion {{{
    Plug 'neoclide/coc.nvim', {'branch': 'release'}
    " to use language server:
    " :CocInstall coc-clangd
    " :CocInstall coc-sh
    " :CocInstall coc-cmake
" }}}

    Plug 'scrooloose/nerdcommenter'
    " nerdcommenter{{{
    let g:NERDSpaceDelims = 1
    let g:NERDTrimTrailingWhitespace = 1
    " }}}
    Plug 'vim-scripts/DoxygenToolkit.vim',{'on':'Dox'}
    let g:load_doxygen_syntax=1
    Plug 'Townk/vim-autoclose'
endif
"}}}

" 1.9 Background jobs {{{
" 自动生成tags文件
Plug 'ludovicchabant/vim-gutentags'
" }}}

" 1.10 Quick access {{{
" 切换头文件 快捷键ts
Plug 'derekwyatt/vim-fswitch',{'for':['c','cpp']}
Plug 'easymotion/vim-easymotion'
"}}}

" 1.11 Snippet {{{
" Track the engine.
Plug 'SirVer/ultisnips'
Plug 'zhangfuwen/vim-snippets'
" Snippets are separated from the engine. Add this if you want them:
" Trigger configuration. You need to change this to something other than <tab> if you use one of the following:
" - https://github.com/Valloric/YouCompleteMe
" - https://github.com/nvim-lua/completion-nvim
"let g:UltiSnipsExpandTrigger="c-y"
"let g:UltiSnipsJumpForwardTrigger="<c-b>"
"let g:UltiSnipsJumpBackwardTrigger="<c-z>"
Plug 'skywind3000/Leaderf-snippet'

" If you want :UltiSnipsEdit to split your window.
"let g:UltiSnipsEditSplit="vertical"
" }}}

" 1.12 Build/Project {{{
Plug 'ilyachur/cmake4vim'
function! Prompt_targets()
    let target_list=cmake4vim#GetAllTargets()
    let submenu = []
    for target in target_list
        call add(submenu, [target, 'CMakeSelectTarget '. target])
    endfor
    call quickui#listbox#open(submenu, {'title':'Select target'})
endfunction
function! Prompt_buildType()
    let build_type_dict=utils#cmake#getCMakeVariants()
    let submenu = []
    for target in keys(build_type_dict)
        call add(submenu, [target, 'CMakeSelectBuildType '. target])
    endfor
    call quickui#listbox#open(submenu, {'title':'Select build type'})
endfunction

function! Run_target()
    let binaryFile=utils#cmake#getBinaryPath()
    if binaryFile == ''
        call Prompt_targets()
    endif
    let binaryFile=utils#cmake#getBinaryPath()
    echo 'path:'. binaryFile
    "call quickui#terminal#open('/bin/sh -c '. binaryFile . '; read', {})
    let opts={'title':'Run'}
    call quickui#terminal#open('bash --init-file <(echo "'. binaryFile .'; echo executed '. binaryFile . '")', opts)
endfunction

" }}}

" Disabled {{{
"Plug 'derekwyatt/vim-protodef',{'for':['c','cpp']}
"Plug 'dbgx/lldb.nvim',{'on':'LLsession','do':':UpdateRemotePlugins'}
" }}}

call plug#end()
"}}}
