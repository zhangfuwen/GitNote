-- init.lua: 完整的Neovim配置文件
-- 整合基础设置与插件管理

-- 0. 环境检测与基础设置
local vim = vim

-- 检测Python环境
if vim.fn.has('python3') == 1 then
    vim.g.pyx = 2
    vim.g.pyxversion = 3
end

-- 1. 插件管理系统 (packer.nvim)
-- 安装packer.nvim（如果尚未安装）
local install_path = vim.fn.stdpath('data') .. '/site/pack/packer/start/packer.nvim'
if vim.fn.empty(vim.fn.glob(install_path)) > 0 then
    vim.fn.system({
        'git', 'clone', '--depth', '1',
        'https://github.com/wbthomason/packer.nvim',
        install_path
    })
    vim.cmd [[packadd packer.nvim]]
end

-- 插件配置
require('packer').startup(function(use)
    -- 插件管理器自身
    use 'wbthomason/packer.nvim'
    
    -- 1.2 Appearance
    use 'vim-airline/vim-airline'
    use 'vim-airline/vim-airline-themes'
    use 'NLKNguyen/papercolor-theme'
    use 'flazz/vim-colorschemes'
    use 'ryanoasis/vim-devicons'
    use 'itchyny/vim-cursorword'
    use 'octol/vim-cpp-enhanced-highlight'
    use 'Yggdroot/indentLine'
    
    -- 1.3 Panels
    use 'scrooloose/nerdtree'
    use 'majutsushi/tagbar'
    
    -- 1.4 PopupTools
    use { 'Yggdroot/LeaderF', run = './install.sh' }
    use 'tiagofumo/vim-nerdtree-syntax-highlight'
    use 'skywind3000/vim-quickui'
    use 'skywind3000/vim-preview'
    
    -- 1.5 VCS
    use 'airblade/vim-gitgutter'
    use 'tpope/vim-fugitive'
    use 'will133/vim-dirdiff'
    use 'gregsexton/gitv'
    
    -- 1.6 Text objects
    use 'kana/vim-textobj-user'
    use 'kana/vim-textobj-indent'
    use 'kana/vim-textobj-syntax'
    use { 'kana/vim-textobj-function', ft = { 'c', 'cpp', 'vim', 'java' } }
    use 'sgur/vim-textobj-parameter'
    
    -- 1.7 Async lint engine
    use 'w0rp/ale'
    
    -- 1.8 Auto completion
    if vim.fn.has('python3') == 1 then
        use { 'neoclide/coc.nvim', branch = 'release' }
        use 'scrooloose/nerdcommenter'
        use { 'vim-scripts/DoxygenToolkit.vim', cmd = 'Dox' }
        use 'Townk/vim-autoclose'
    end
    
    -- 1.9 Background jobs
    use 'ludovicchabant/vim-gutentags'
    
    -- 1.10 Quick access
    use { 'derekwyatt/vim-fswitch', ft = { 'c', 'cpp' } }
    use 'easymotion/vim-easymotion'
    
    -- 1.11 Snippet
    use 'SirVer/ultisnips'
    use 'zhangfuwen/vim-snippets'
    use 'skywind3000/Leaderf-snippet'
    
    -- 1.12 Build/Project
    use 'ilyachur/cmake4vim'
    
    -- 1.13 plantuml
    use 'tyru/open-browser.vim'
    use 'aklt/plantuml-syntax'
    use 'weirongxu/plantuml-previewer.vim'
    
    -- 1.14 ctrlsf
    use 'dyng/ctrlsf.vim'
end)

-- 自动编译插件配置
vim.cmd([[
augroup packer_user_config
    autocmd!
    autocmd BufWritePost init.lua source <afile> | PackerCompile
augroup end
]])

-- 2. 自定义脚本功能
-- 定义FindAll函数
vim.api.nvim_create_user_command('FindAllHere', function()
    local p = vim.fn.input('Enter pattern to search in this file:')
    if p ~= '' then
        local success, err = pcall(vim.cmd, 'vimgrep "' .. p .. '" %|copen')
        if success then
            vim.cmd('cope')
        else
            vim.notify("Not anything found")
        end
    end
end, {})

-- 定义GREP命令
vim.api.nvim_create_user_command('GREP', function()
    local cword = vim.fn.expand('<cword>')
    local filename = vim.fn.expand('%')
    vim.cmd('vimgrep ' .. cword .. ' ' .. filename .. '|copen|cc')
end, {})

-- 3. 核心功能配置
-- 3.1 基本设置
vim.opt.mouse = 'a'
vim.opt.filetype = 'plugin'
vim.opt.compatible = false
vim.opt.filetype = 'on'
vim.opt.syntax = 'enable'
vim.opt.backspace = 'indent,eol,start'
vim.opt.cursorline = true

-- 颜色主题
if vim.fn.empty(vim.fn.stdpath('data').."/pack/packer/start/papercolor-theme/colors/PaperColor.vim") == 0 then
    vim.cmd('colo PaperColor')
end

-- 3.2 折叠设置
vim.opt.foldenable = false
vim.api.nvim_create_autocmd('FileType', {
    pattern = {'c', 'cpp', 'perl'},
    command = 'set foldmethod=syntax'
})
vim.api.nvim_create_autocmd('FileType', {
    pattern = 'python',
    command = 'set foldmethod=indent'
})
vim.api.nvim_create_autocmd('FileType', {
    pattern = 'vim',
    command = 'set foldmethod=marker | set nowrap'
})

-- 3.3 缩进设置
vim.opt.expandtab = true
vim.opt.tabstop = 4
vim.opt.shiftwidth = 4
vim.opt.softtabstop = 4
vim.opt.autoindent = true
vim.opt.smartindent = true

-- 3.4 滚动设置
vim.opt.scrolloff = 4
vim.opt.sidescrolloff = 7

-- 3.5 编码设置
vim.opt.helplang = 'cn'
vim.opt.encoding = 'utf-8'
vim.opt.termencoding = 'utf-8'
vim.opt.fileencodings = 'utf-8,ucs-bom,cp936,gb18030,latin1'
vim.opt.fileencoding = 'utf-8'
vim.opt.fileformat = 'unix'

-- 3.6 搜索设置
vim.opt.hlsearch = true
vim.opt.incsearch = true
vim.opt.ignorecase = true

-- 3.7 补全设置
vim.keymap.set('i', '<Tab>', function()
    return vim.fn.pumvisible() == 1 and '<C-n>' or '<Tab>'
end, {expr = true, silent = true})
vim.keymap.set('i', '<S-Tab>', function()
    return vim.fn.pumvisible() == 1 and '<C-p>' or '<S-Tab>'
end, {expr = true, silent = true})

-- 3.8 ctags设置
vim.opt.tags = './.tags;,.tags,tags'
vim.g.gutentags_project_root = {'.root', '.svn', '.git', '.hg', '.project'}
vim.g.gutentags_ctags_tagfile = '.tags'
local s_vim_tags = vim.fn.expand('~/.cache/tags')
vim.g.gutentags_cache_dir = s_vim_tags

-- 配置ctags参数
vim.g.gutentags_ctags_extra_args = {'--fields=+niazS', '--extra=+q'}
table.insert(vim.g.gutentags_ctags_extra_args, '--c++-kinds=+px')
table.insert(vim.g.gutentags_ctags_extra_args, '--c-kinds=+px')

-- 检测缓存目录
if vim.fn.isdirectory(s_vim_tags) == 0 then
    vim.fn.system({'mkdir', '-p', s_vim_tags})
end

-- 3.9 nerdtree设置
vim.g.NERDTreeQuitOnOpen = 1

-- 3.10 其他设置
vim.opt.number = true
vim.opt.autoread = true
vim.opt.showmatch = true
vim.opt.laststatus = 2
--vim.opt.t_Co = 256
vim.opt.completeopt = 'menu,menuone'
vim.opt.background = 'light'
vim.opt.wildmenu = true

-- 4. QuickMenu设置（需要vim-quickui插件支持）
if vim.fn.empty(vim.fn.stdpath('data').."/pack/packer/start/quickui.vim/README.md") == 0 then
    -- 清除所有菜单
    vim.cmd('call quickui#menu#reset()')
    
    -- 安装菜单（保留Vimscript调用格式）
    vim.g.quickui_border_style = 2
    vim.cmd([[call quickui#menu#install('&Find', [
        \ ["Switch &Header/Source\tta", 'FSHere'],
        \ ["Search &In This File\tts", 'silent! FindAllHere' ],
        \ ["--", '' ],
        \ ["E&xit\tAlt+x", 'echo 6' ],
        \])
        ]])
    
    -- 安装View菜单
    vim.cmd([[call quickui#menu#install('&View', [
        \ ["Open one fold here\tzo", 'normal zo'],
        \ ["&Open all fold here\tzO", 'normal zO'],
        \ ["close one fold here\tzc", 'normal zc'],
        \ ["&Close all fold here\tzC", 'normal zC'],
        \ ["Open all fold\tzM", 'normal zM'],
        \ ["Close all fold\tzR", 'normal zR'],
        \])
        ]])
    
    -- 安装Quickfix菜单
    vim.cmd([[call quickui#menu#install('&Quickfix', [
        \ ["&Open\t copen", 'copen' ],
        \ ["&Close\t cclose", 'ccl' ],
        \ ["&Next\t cnext", 'cnext' ],
        \ ["&Prev\t cprev", 'cprev' ],
        \ ["&First\t cfirst", 'cfirst' ],
        \ ["&Last\t clast", 'clast' ],
        \ ["Olde&r\t colder", 'colder' ],
        \ ["Ne&wer\t cnewer", 'cnewer' ],
        \])
        ]])
    
    -- 安装LeaderF菜单
    vim.cmd([[call quickui#menu#install('Leader&f', [
        \ ["&File\t file", 'Leaderf file' ],
        \ ["&Tag\t tag", 'Leaderf tag' ],
        \ ["&Snippet\t snippet", 'Leaderf snippet' ],
        \ ["&Grep\t search", 'Leaderf rg' ],
        \ ["Rg &Interactive", 'LeaderfRgInteractive' ],
        \ ["Grep search &recall", 'LeaderfRgRecall' ],
        \ ["F&unction\t function", 'Leaderf function' ],
        \ ["&Buffers", 'Leaderf buffer' ],
        \])
        ]])
    vim.cmd([[
call quickui#menu#install('&Locationlist', [
        \ [ "&Open\t lopen", 'lopen' ],
        \ [ "&Close\t lclose", 'lcl' ],
        \ [ "&Next\t lnext", 'lnext' ],
        \ [ "&Prev\t lprev", 'lprev' ],
        \ [ "&First\t lfirst", 'lfirst' ],
        \ [ "&Last\t llast", 'llast' ],
        \ [ "Olde&r\t lolder", 'lolder' ],
        \ [ "Ne&wer\t lnewer", 'lnewer' ],
        \ ])

call quickui#menu#install('&Coc', [
        \ [ "List &diagnostics\t ", 'CocList diagnostics' ],
        \ [ "List &extentions\t ", 'CocList extentions' ],
        \ [ "List &commands\t ", 'CocList commands' ],
        \ [ "List &outline\t ", 'CocList outline' ],
        \ [ "List &symbols\t ", 'CocList symbols' ],
        \ [ "List &resume\t ", 'CocListResume' ],
        \ [ "&Next\t ", 'CocNext' ],
        \ [ "&Prev\t ", 'CocPrev' ],
        \ [ "For&mat", "Format" ],
        \ [ "Fo&ld", "Fold" ],
        \ [ "Rearra&ge imports ", "OR" ],
        \ [ "Ren&ame", "<Plug>(coc-rename)" ],
        \ ])

let g:cmake_compile_commands=1
let g:cmake_compile_commands_link='.'
call quickui#menu#install('&CMake', [
        \ ['&Generate','CMake'],
        \ ['&Build','CMakeBuild'],
        \ ['&Test','CTest'],
        \ ['&CTest!','CTest!'],
        \ ['&Info','CMakeInfo'],
        \ ['&Select Target', 'call Prompt_targets()'],
        \ ['Select Build T&ype', 'call Prompt_buildType()'],
        \ ['&Run','call Run_target()'],
        \ ['R&un!','CMakeRun!'],
        \ ['C&lean','CMakeClean'],
        \ ['Res&et','CMakeReset'],
        \ ['Reset&Relo&ad','CMakeResetAndReload' ],
        \ ])

call quickui#menu#install('&Preview', [
        \ [ "&Close\t pc", 'pc' ],
        \ [ "&Search\t ps", 'ps' ],
        \ [ "&Edit\t ped", 'ped' ],
        \ [ "&Jump\t ptjump", 'ptjump' ],
        \ [ "&Tag\t ptag", 'ptag' ],
        \ ])

call quickui#menu#install('&Git', [
        \ [ "&Status\t G", 'G' ],
        \ [ "&Llog\t Gllog", 'Gllog' ],
        \ [ "&Clog\t Gclog", 'Gclog' ],
        \ ])

call quickui#menu#install('P&ython', [
        \ [ "&Run this file with python3", ":exec '!python3' shellescape(@%, 1)" ],
        \ ])

" list
call quickui#menu#install('&List', [
        \ [ "&Buffers", "call quickui#tools#list_buffer('e')" ],
        \ [ "&Functions", "call quickui#tools#list_function()" ],
        \ ])
" items containing tips, tips will display in the cmdline
call quickui#menu#install('&Open', [
        \ [ '&Terminal', "call quickui#terminal#open('bash', {'title':'terminal'})", 'help 1' ],
        \ ])
"            \ [ '&Terminal', "call quickui#terminal#open('bash', {'w':60, 'h':8, 'callback':'TermExit', 'title':'terminal'})", 'help 1' ],

" script inside %{...} will be evaluated and expanded in the string
call quickui#menu#install("&Option", [
        \ ['Set &Spell %{&spell? "Off":"On"}', 'set spell!'],
        \ ['Set &Cursor Line %{&cursorline? "Off":"On"}', 'set cursorline!'],
        \ ['Set &Paste %{&paste? "Off":"On"}', 'set paste!'],
        \ ])

    ]])
    
    -- 注册HELP菜单
    vim.cmd([[call quickui#menu#install('H&elp', [
        \ ["&Cheatsheet", 'help index', ''],
        \ ['T&ips', 'help tips', ''],
        \ ['--',''],
        \ ["&Tutorial", 'help tutor', ''],
        \ ['&Quick Reference', 'help quickref', ''],
        \ ['&Summary', 'help summary', ''],
        \ ], 10000)
        ]])
    
    -- 映射快捷键
    vim.keymap.set('n', 'to', ':call quickui#menu#open()<CR>')
    vim.g.quickui_show_tip = 1
    
    -- 定义TermExit函数
    vim.api.nvim_create_user_command('TermExit', function(args)
        vim.notify("terminal exit code: " .. args[1])
    end, {})
    
    -- 定义上下文菜单
    vim.g.context_menu_k = {
        ["&Help Keyword\t\\ch"] = 'echo expand("<cword>")',
        ["&Signature\t\\cs"] = 'echo 101',
        ["-"] = "",
        ["Find in &File\t\\cx"] = 'exec "/" . expand("<cword>")',
        ["Find in &Project\t\\cp"] = 'exec "vimgrep " . expand("<cword>") . "*"',
        ["Find in &Defintion\t\\cd"] = 'YcmCompleter GotoDefinition',
        ["Search &References\t\\cr"] = 'YcmCompleter GoToReferences',
        ["-"] = "",
        ["&Documentation\t\\cm"] = 'exec "PyDoc " . expand("<cword>")',
    }
    
    -- 映射空格键显示上下文菜单
    vim.keymap.set('n', '<space>', ':call quickui#tools#clever_context("k", g:context_menu_k, {})<cr>', {silent = true})
    
    -- Git上下文菜单
    vim.g.context_menu_git = {
        ["&Stage (add)\ts"] = 'exec "normal s"',
        ["&Unstage (reset)\tu"] = 'exec "normal u"',
        ["&Toggle stage/unstage\t-"] = 'exec "normal -"',
        ["Unstage &Everything\tU"] = 'exec "normal U"',
        ["D&iscard change\tX"] = 'exec "normal X"',
        ["--"]='',
        ["Inline &Diff\t="] = 'exec "normal ="',
        ["Diff Split\tdd"] = 'exec "normal dd"',
        ["Diff Horizontal\tdh"] = 'exec "normal dh"',
        ["Diff &Vertical\tdv"] = 'exec "normal dv"',
        ["--"]='',
        ["&Open File\t<CR>"] = 'exec "normal o<cr>"',
        ["Open in New Split\to"] = 'exec "normal o"',
        ["Open in New Vsplit\tgO"] = 'exec "normal gO"',
        ["Open in New Tab\tO"] = 'exec "normal O"',
        ["Open in &Preview\tp"] = 'exec "normal p"',
        ["--"]='',
        ["&0. Commit"] = 'Git commit',
        ["&1. Push"] = 'Git push',
        ["&2. Pull"] = 'Git pull',
    }
    
    -- 设置fugitive文件类型的映射
    local setup_fugitive = function()
        vim.keymap.set('n', '<space>', ':call quickui#tools#clever_context("g", g:context_menu_git, {})<cr>', {silent = true, buffer = true})
    end
    
    -- 创建自动命令组
    vim.api.nvim_create_augroup('MenuEvents', {clear = true})
    vim.api.nvim_create_autocmd('FileType', {
        group = 'MenuEvents',
        pattern = 'fugitive',
        callback = setup_fugitive
    })
end

-- 5. 快捷键映射
vim.keymap.set('n', 'tt', ':NERDTreeToggle<CR>')
vim.keymap.set('n', 'tl', ':TagbarToggle<CR>')
vim.keymap.set('n', 't/', ':silent! FindAllHere<CR>')
vim.keymap.set('n', 'ta', ':FSHere<CR>') -- 头文件切换
vim.keymap.set('n', 'ts', ':Leaderf rg -- "<C-r><C-w>"<CR>')
vim.keymap.set('n', 'tv', ':PreviewTag<CR>')

-- easy motion映射
vim.keymap.set({'n', 'x'}, '<Leader>f', '<Plug>(easymotion-bd-f)')
vim.keymap.set('n', '<Leader>f', '<Plug>(easymotion-overwin-f)')
vim.keymap.set('n', 's', '<Plug>(easymotion-overwin-f2)')
vim.keymap.set({'n', 'x'}, '<C-L>', '<Plug>(easymotion-bd-jk)')
vim.keymap.set('n', '<C-L>', '<Plug>(easymotion-overwin-line)')
vim.keymap.set({'n', 'x'}, '<C-L>u', '<Plug>(easymotion-bd-w)')
vim.keymap.set('n', '<C-L>u', '<Plug>(easymotion-overwin-w)')

-- F6键映射
vim.keymap.set('v', '<F6>', ':w !bash<CR>')
vim.keymap.set('n', '<C-_>', '<leader>c<space>')
vim.keymap.set('n', '<F3>', ':call quickui#tools#preview_tag(\'\')<cr>')

-- 6. CMake相关函数
--local cmake4vim = require('cmake4vim')

--function Prompt_targets()
--    local target_list = vim.fn.cmake4vim#GetAllTargets()
--    local submenu = {}
--    for _, target in ipairs(target_list) do
--        table.insert(submenu, {target, 'CMakeSelectTarget ' .. target})
--    end
--    vim.fn['quickui#listbox#open'](submenu, {title = 'Select target'})
--end

function Prompt_buildType()
    local build_type_dict = require('utils.cmake').getCMakeVariants()
    local submenu = {}
    for target, _ in pairs(build_type_dict) do
        table.insert(submenu, {target, 'CMakeSelectBuildType ' .. target})
    end
    vim.fn['quickui#listbox#open'](submenu, {title = 'Select build type'})
end

function Run_target()
    local binaryFile = require('utils.cmake').getBinaryPath()
    if binaryFile == '' then
        Prompt_targets()
        binaryFile = require('utils.cmake').getBinaryPath()
    end
    print('path:' .. binaryFile)
    local opts = {title = 'Run'}
    vim.fn['quickui#terminal#open']('bash --init-file <(echo "' .. binaryFile .. '; echo executed ' .. binaryFile .. '")', opts)
end

-- 7. 非GUI环境设置
if vim.fn.has("gui_running") == 0 then
    vim.cmd('source $VIMRUNTIME/menu.vim')
    vim.opt.wildmenu = true
    vim.opt.cpoptions:remove('<')
--    vim.opt.wildcharm = '<C-Z>'
    vim.keymap.set('n', '<F4>', ':emenu <C-Z><CR>')
end

-- 8. ALE配置
vim.g.ale_linters_explicit = 1
vim.g.ale_completion_delay = 500
vim.g.ale_echo_delay = 20
vim.g.ale_lint_delay = 500
vim.g.ale_echo_msg_format = '[%linter%] %code: %%s'
vim.g.ale_lint_on_text_changed = 'normal'
vim.g.ale_lint_on_insert_leave = 1
--vim.g.airline#extensions#ale#enabled = 1

vim.g.ale_c_gcc_options = '-Wall -O2 -std=c99'
vim.g.ale_cpp_gcc_options = '-Wall -O2 -std=c++14'
vim.g.ale_c_cppcheck_options = ''
vim.g.ale_cpp_cppcheck_options = ''

-- 9. NerdCommenter配置
vim.g.NERDSpaceDelims = 1
vim.g.NERDTrimTrailingWhitespace = 1

-- 10. Doxygen配置
vim.g.load_doxygen_syntax = 1

-- 11. 文本对象映射（示例）
-- 可以在这里添加更多文本对象映射

-- 12. 启动时执行的命令
vim.cmd([[
" 初始化LeaderF设置
let g:Lf_WindowPosition = 'popup'

" 初始化UltiSnips设置
" let g:UltiSnipsExpandTrigger = "c-y"
" let g:UltiSnipsJumpForwardTrigger = "<c-b>"
" let g:UltiSnipsJumpBackwardTrigger = "<c-z>"

" 初始化CMake4Vim设置
let g:cmake_compile_commands = 1
let g:cmake_compile_commands_link = '.'
]])
