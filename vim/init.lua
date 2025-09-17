-- Function: download_if_missing(url, dest_path)
-- Downloads a file from `url` to `dest_path` only if it doesn't exist.
-- Uses curl or wget (fallback).
-- Creates parent directories if missing.
--
-- Example:
--   download_if_missing(
--     "https://raw.githubusercontent.com/yourname/config/main/plugins.lua",
--     vim.fn.stdpath("config") .. "/lua/plugins.lua"
--   )
local function download_if_missing(url, dest_path)
    -- Create directory if it doesn't exist
    local dir = vim.fn.fnamemodify(dest_path, ":h")
    if not vim.loop.fs_stat(dir) then
        vim.fn.mkdir(dir, "p")
    end

    -- Check if file already exists
    if vim.loop.fs_stat(dest_path) then
        print("📄 " .. dest_path .. " already exists. Skipping download.")
        return
    end

    print("📥 Downloading from: " .. url)
    print("💾 Saving to: " .. dest_path)

    -- Try curl first, fallback to wget
    local cmd = string.format("curl -s -o %q %q", dest_path, url)
    local success = pcall(vim.fn.system, cmd)

    if not success then
        cmd = string.format("wget -O %q %q", dest_path, url)
        success = pcall(vim.fn.system, cmd)
    end

    if success then
        print("✅ Successfully downloaded: " .. dest_path)
    else
        error("❌ Failed to download from " .. url .. ". Please check internet connection or URL.")
    end
end

-- Example: Download plugins.lua from GitHub
download_if_missing(
    "https://raw.githubusercontent.com/zhangfuwen/GitNote/master/vim/plugins.lua",
    vim.fn.stdpath("config") .. "/lua/plugins.lua"
)

-- Install lazy.nvim if missing
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
    vim.fn.system({
        "git", "clone", "--filter=blob:none", "--branch=stable",
        "https://github.com/folke/lazy.nvim.git", lazypath
    })
end
vim.opt.rtp:prepend(lazypath)

-- Load plugins
vim.g.mapleader = ","
require("lazy").setup("plugins")


-- 0. 环境检测与基础设置
local vim = vim

-- 检测Python环境
if vim.fn.has('python3') == 1 then
    vim.g.pyx = 2
    vim.g.pyxversion = 3
end


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
--vim.opt.mousemodel=extend

-- 颜色主题
-- if vim.fn.empty(vim.fn.stdpath('data') .. "/pack/packer/start/papercolor-theme/colors/PaperColor.vim") == 0 then
--     vim.cmd('colo PaperColor')
-- end
    vim.cmd('colo tokyonight')

-- 3.2 折叠设置
vim.opt.foldenable = false
vim.api.nvim_create_autocmd('FileType', {
    pattern = { 'c', 'cpp', 'perl' },
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
--vim.opt.termencoding = 'utf-8'
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
end, { expr = true, silent = true })
vim.keymap.set('i', '<S-Tab>', function()
    return vim.fn.pumvisible() == 1 and '<C-p>' or '<S-Tab>'
end, { expr = true, silent = true })

-- 3.8 ctags设置
vim.opt.tags = './.tags;,.tags,tags'
vim.g.gutentags_project_root = { '.root', '.svn', '.git', '.hg', '.project' }
vim.g.gutentags_ctags_tagfile = '.tags'
local s_vim_tags = vim.fn.expand('~/.cache/tags')
vim.g.gutentags_cache_dir = s_vim_tags

-- 配置ctags参数
vim.g.gutentags_ctags_extra_args = { '--fields=+niazS', '--extra=+q' }
table.insert(vim.g.gutentags_ctags_extra_args, '--c++-kinds=+px')
table.insert(vim.g.gutentags_ctags_extra_args, '--c-kinds=+px')

-- 检测缓存目录
if vim.fn.isdirectory(s_vim_tags) == 0 then
    vim.fn.system({ 'mkdir', '-p', s_vim_tags })
end


-- 3.10 其他设置
vim.opt.number = true
vim.opt.autoread = true
vim.opt.showmatch = true
vim.opt.laststatus = 2
--vim.opt.t_Co = 256
vim.opt.completeopt = 'menu,menuone'
vim.opt.background = 'light'
vim.opt.wildmenu = true



-- 安装菜单（保留Vimscript调用格式）

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
vim.keymap.set('n', '<space>', ':call quickui#tools#clever_context("k", g:context_menu_k, {})<cr>', { silent = true })

-- Git上下文菜单
vim.g.context_menu_git = {
    ["&Stage (add)\ts"] = 'exec "normal s"',
    ["&Unstage (reset)\tu"] = 'exec "normal u"',
    ["&Toggle stage/unstage\t-"] = 'exec "normal -"',
    ["Unstage &Everything\tU"] = 'exec "normal U"',
    ["D&iscard change\tX"] = 'exec "normal X"',
    ["--"] = '',
    ["Inline &Diff\t="] = 'exec "normal ="',
    ["Diff Split\tdd"] = 'exec "normal dd"',
    ["Diff Horizontal\tdh"] = 'exec "normal dh"',
    ["Diff &Vertical\tdv"] = 'exec "normal dv"',
    ["--"] = '',
    ["&Open File\t<CR>"] = 'exec "normal o<cr>"',
    ["Open in New Split\to"] = 'exec "normal o"',
    ["Open in New Vsplit\tgO"] = 'exec "normal gO"',
    ["Open in New Tab\tO"] = 'exec "normal O"',
    ["Open in &Preview\tp"] = 'exec "normal p"',
    ["--"] = '',
    ["&0. Commit"] = 'Git commit',
    ["&1. Push"] = 'Git push',
    ["&2. Pull"] = 'Git pull',
}

-- 设置fugitive文件类型的映射
local setup_fugitive = function()
    vim.keymap.set('n', '<space>', ':call quickui#tools#clever_context("g", g:context_menu_git, {})<cr>',
        { silent = true, buffer = true })
end

-- 创建自动命令组
vim.api.nvim_create_augroup('MenuEvents', { clear = true })
vim.api.nvim_create_autocmd('FileType', {
    group = 'MenuEvents',
    pattern = 'fugitive',
    callback = setup_fugitive
})

-- 5. 快捷键映射
vim.keymap.set('n', 'ta', ':FSHere<CR>') -- 头文件切换

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
        table.insert(submenu, { target, 'CMakeSelectBuildType ' .. target })
    end
    vim.fn['quickui#listbox#open'](submenu, { title = 'Select build type' })
end

function Run_target()
    local binaryFile = require('utils.cmake').getBinaryPath()
    if binaryFile == '' then
        Prompt_targets()
        binaryFile = require('utils.cmake').getBinaryPath()
    end
    print('path:' .. binaryFile)
    local opts = { title = 'Run' }
    vim.fn['quickui#terminal#open'](
    'bash --init-file <(echo "' .. binaryFile .. '; echo executed ' .. binaryFile .. '")', opts)
end

-- 7. 非GUI环境设置
if vim.fn.has("gui_running") == 0 then
    vim.cmd('source $VIMRUNTIME/menu.vim')
    vim.opt.wildmenu = true
    vim.opt.cpoptions:remove('<')
    --    vim.opt.wildcharm = '<C-Z>'
    vim.keymap.set('n', '<F4>', ':emenu <C-Z><CR>')
end

-- 10. Doxygen配置
vim.g.load_doxygen_syntax = 1

-- 11. 文本对象映射（示例）
-- 可以在这里添加更多文本对象映射

-- 12. 启动时执行的命令
vim.cmd([[
    " 初始化CMake4Vim设置
    let g:cmake_compile_commands = 1
    let g:cmake_compile_commands_link = '.'
    ]])



-- init.lua

vim.cmd([[
    nnoremenu PopUp.SplitRight :vsplit term://zsh \| :normal i<CR><CR>
    nnoremenu PopUp.SplitDown :split term://zsh \| :normal i<CR><CR>
    nnoremenu PopUp.SplitLeft :vertical leftabove vsplit term://zsh \| :normal i <CR>
    nnoremenu PopUp.SplitUp :leftabove split term://zsh \| :normal i <CR>
    nnoremenu PopUp.TempTerminal :call quickui#terminal#open('bash', {'w':80, 'h':32, 'callback':'', 'title':'terminal'})<CR>
    nnoremenu PopUp.Close\ Terminal :bdelete!<CR>
    ]])
