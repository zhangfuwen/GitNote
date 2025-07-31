-- ~/.config/nvim/lua/plugins.lua
quickui_menu_cmds = [[
call quickui#menu#reset()
call quickui#menu#install('&Find', [
\ ["Switch &Header/Source\tta", 'FSHere'],
\ ["Search &In This File\tts", 'silent! FindAllHere' ],
\ ["--", '' ],
\ ["&Find", 'FzfLua lines' ],
\ ["&Find in files", 'Leaderf file' ],
\ ["&Symbols", 'FzfLua lsp_workspace_symbols' ],
\ ["Fzf refs", 'FzfLua lsp_references' ],
\ ["&Tag\t tag", 'Leaderf tag' ],
\ ["S&nippet\t snippet", 'Leaderf snippet' ],
\ ["&Grep\t search", 'Leaderf rg' ],
\ ["Rg &Interactive", 'LeaderfRgInteractive' ],
\ ["Grep search &recall", 'LeaderfRgRecall' ],
\ ["F&unction\t function", 'Leaderf function' ],
\ ["&Buffers", 'Leaderf buffer' ],
\ ["E&xit\tAlt+x", 'echo 6' ],
\])
call quickui#menu#install('&Code', [
\ ["Format file", ':lua vim.lsp.buf.format()'],
\ ["Comment lines", ':lua vim.lsp.buf.execute_command({ command = "editor.action.commentLine" })'],
\ ["Cod all fold\tzR", 'normal zR'],
\])
call quickui#menu#install('&View', [
\ ["Open one fold here\tzo", 'normal zo'],
\ ["&Open all fold here\tzO", 'normal zO'],
\ ["close one fold here\tzc", 'normal zc'],
\ ["&Close all fold here\tzC", 'normal zC'],
\ ["Open all fold\tzM", 'normal zM'],
\ ["Close all fold\tzR", 'normal zR'],
\])
call quickui#menu#install('&Quickfix', [
\ ["&Open\t copen", 'copen' ],
\ ["&Close\t cclose", 'ccl' ],
\ ["&Next\t cnext", 'cnext' ],
\ ["&Prev\t cprev", 'cprev' ],
\ ["&First\t cfirst", 'cfirst' ],
\ ["&Last\t clast", 'clast' ],
\ ["Olde&r\t colder", 'colder' ],
\ ["Ne&wer\t cnewer", 'cnewer' ],
\])

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

call quickui#menu#install('&Run', [
\ [ "&Run this file with python3", ":exec '!python3' shellescape(@%, 1)" ],
\ [ "&Run this file with bash", ":exec '!bash' shellescape(@%, 1)" ],
\ ])

" list
call quickui#menu#install('&List', [
\ [ "&Buffers", "call quickui#tools#list_buffer('e')" ],
\ [ "&Functions", "call quickui#tools#list_function()" ],
\ ])
" items containing tips, tips will display in the cmdline
call quickui#menu#install('&Terminal', [
\ [ '&Terminal', "call quickui#terminal#open('bash', {'title':'terminal'})", 'help 1' ],
\ ])
"            \ [ '&Terminal', "call quickui#terminal#open('bash', {'w':60, 'h':8, 'callback':'TermExit', 'title':'terminal'})", 'help 1' ],

" script inside %{...} will be evaluated and expanded in the string
call quickui#menu#install("&Option", [
\ ['Set &Spell %{&spell? "Off":"On"}', 'set spell!'],
\ ['Set &Cursor Line %{&cursorline? "Off":"On"}', 'set cursorline!'],
\ ['Set &Paste %{&paste? "Off":"On"}', 'set paste!'],
\ ])


call quickui#menu#install('&Help', [
\ ["Edit init.lua", 'e ~/.config/nvim/init.lua' ],
\ ["Edit plugins.lua", 'e ~/.config/nvim/lua/plugins.lua' ],
\ ["&Lazy", 'Lazy', ''],
\ ["&Mason", 'Mason', ''],
\ ["&Cheatsheet", 'help index', ''],
\ ['T&ips', 'help tips', ''],
\ ['--',''],
\ ["&Tutorial", 'help tutor', ''],
\ ['&Quick Reference', 'help quickref', ''],
\ ['&Summary', 'help summary', ''],
\ ], 10000)

]]
return {
    ----------------------------------------------------------------------
    -- 🧩 PLUGIN MANAGER
    ----------------------------------------------------------------------
    {
        "folke/lazy.nvim",
        version = false, -- auto-update
    },

    ----------------------------------------------------------------------
    -- 🌐 LSP & LANGUAGE TOOLS
    ----------------------------------------------------------------------
    {
        "neovim/nvim-lspconfig",
        event = "BufReadPre",
        dependencies = {
            { "williamboman/mason.nvim",          config = true },
            { "williamboman/mason-lspconfig.nvim" },
        },
        config = function()
            require("mason").setup()
            require("mason-lspconfig").setup({
                ensure_installed = { "pyright", "clangd" }, -- customize as needed
            })
            --      require("lspconfig").setup()
        end,
    },

    ----------------------------------------------------------------------
    -- 🎨 APPEARANCE
    ----------------------------------------------------------------------
    { "ryanoasis/vim-devicons",           event = "VeryLazy" },
    { "vim-airline/vim-airline",          event = "VeryLazy" },
    { "vim-airline/vim-airline-themes",   after = "vim-airline" },
    { "NLKNguyen/papercolor-theme",       lazy = false },
    { "flazz/vim-colorschemes",           lazy = false },
    { "itchyny/vim-cursorword",           event = "BufReadPre" },
    { "octol/vim-cpp-enhanced-highlight", ft = "cpp" },
    { "Yggdroot/indentLine",              event = "BufReadPre" },

    ----------------------------------------------------------------------
    -- 📁 FILE & PROJECT MANAGEMENT
    ----------------------------------------------------------------------
    {
        "scrooloose/nerdtree",
        cmd = "NERDTreeToggle",
        keys = { { "<leader>n", "<cmd>NERDTreeToggle<cr>", desc = "Toggle NERDTree" } },
    },
    {
        "tiagofumo/vim-nerdtree-syntax-highlight",
        ft = "nerdtree",
        opts = {},
    },
    {
        "majutsushi/tagbar",
        cmd = "TagbarToggle",
        keys = { { "tb", ":TagbarToggle", desc = "Toggle Tagbar" } },
    },

    ----------------------------------------------------------------------
    -- 🔍 SEARCH & FIND
    ----------------------------------------------------------------------
    {
        "Yggdroot/LeaderF",
        build = "./install.sh",
        cmd = "Leaderf",
        keys = { { "<leader>f", "<cmd>Leaderf<cr>", desc = "LeaderF" } },
    },
    {
        "skywind3000/vim-quickui",
        event = "VeryLazy",
        config = function()
            vim.g.quickui_border_style = 2
            vim.keymap.set('n', 'to', ':call quickui#menu#open()<CR>')
            vim.g.quickui_show_tip = 1
            vim.cmd(quickui_menu_cmds)
        end,
    },
    {
        "skywind3000/vim-preview",
        event = "VeryLazy",
    },
    -- {
        -- "dyng/ctrlsf.vim",
        -- cmd = "CtrlSF",
        -- keys = { { "<leader>fs", "<cmd>CtrlSF ", desc = "Search in files" } },
    -- },

    ----------------------------------------------------------------------
    -- 🛠 TEXT OBJECTS & EDITING
    ----------------------------------------------------------------------
    { "kana/vim-textobj-user",      event = "VeryLazy" },
    --  { "kana/vim-textobj-indent",        event = "VeryLazy" },
    --  { "kana/vim-textobj-syntax",        event = "VeryLazy" },
    {
        "kana/vim-textobj-function",
        ft = { "c", "cpp", "vim", "java" },
    },
    { "sgur/vim-textobj-parameter", ft = { "c", "cpp", "go", "rust" } },

    ----------------------------------------------------------------------
    -- 💬 COMMENTS & AUTO-CLOSING
    ----------------------------------------------------------------------
    -- { "scrooloose/nerdcommenter",   event = "BufReadPre" },
    {
        "Townk/vim-autoclose",
        event = "InsertEnter",
        --    keys = { { "(", "()", mode = "i" }, { "[", "[]", mode = "i" }, { "\"", "\"\"", mode = "i" } },
    },

    ----------------------------------------------------------------------
    -- 📄 DOXYGEN & DOCUMENTATION
    ----------------------------------------------------------------------
    {
        "vim-scripts/DoxygenToolkit.vim",
        cmd = "Dox",
        config = function()
            vim.g.doxygen_enhanced_color = 1
        end,
    },

    ----------------------------------------------------------------------
    -- 🎯 EASYMOTION & QUICK ACCESS
    ----------------------------------------------------------------------
    {
        "easymotion/vim-easymotion",
        keys = {
            { "<leader>j", "<Plug>(easymotion-j)", mode = "n" },
            { "<leader>k", "<Plug>(easymotion-k)", mode = "n" },
        },
    },
    {
        "derekwyatt/vim-fswitch",
        ft = { "c", "cpp" },
    },

    ----------------------------------------------------------------------
    -- 🧩 SNIPPETS
    ----------------------------------------------------------------------
    {
        "L3MON4D3/LuaSnip",
        event = "InsertEnter",
        dependencies = {
            "saadparwaiz1/cmp_luasnip",
            "rafamadriz/friendly-snippets",
        },
        opts = {
            history = true,
            delete_check_events = "InsertLeave",
        },
        config = function(_, opts)
            require("luasnip").setup(opts)
            require("luasnip.loaders.from_vscode").lazy_load()
        end,
    },
    { "zhangfuwen/vim-snippets", after = "LuaSnip" },
    { "SirVer/ultisnips",        ft = { "c", "cpp", "python" } },

    ----------------------------------------------------------------------
    -- 🔍 FZF & COMPLETION
    ----------------------------------------------------------------------
    {
        "ibhagwan/fzf-lua",
        cmd = "FzfLua",
        keys = { { "<leader>ff", "<cmd>FzfLua files<cr>", desc = "Find files" } },
        opts = {},
    },
    {
        "hrsh7th/nvim-cmp",
        event = "InsertEnter",
        dependencies = {
            "hrsh7th/cmp-nvim-lsp",
            "L3MON4D3/LuaSnip",
            "saadparwaiz1/cmp_luasnip",
        },
        config = function()
            local cmp = require("cmp")
            cmp.setup({
                snippet = {
                    expand = function(args)
                        require("luasnip").lsp_expand(args.body)
                    end,
                },
                mapping = cmp.mapping.preset.insert({
                    ["<C-Space>"] = cmp.mapping.complete(),
                    ["<CR>"] = cmp.mapping.confirm({ select = false }),
                }),
                sources = cmp.config.sources({
                    { name = "nvim_lsp" },
                    { name = "luasnip" },
                }, {
                    { name = "buffer" },
                }),
            })
        end,
    },

    ----------------------------------------------------------------------
    -- 🐞 GIT & VCS
    ----------------------------------------------------------------------
    { "airblade/vim-gitgutter",           event = "BufReadPre" },
    { "tpope/vim-fugitive",               cmd = { "Git", "G" } },
    { "will133/vim-dirdiff",              cmd = "DirDiff" },
    { "gregsexton/gitv",                  cmd = "Gitv" },

    ----------------------------------------------------------------------
    -- 🧱 BUILD / PROJECT TOOLS
    ----------------------------------------------------------------------
    { "ilyachur/cmake4vim",               ft = "cmake" },

    ----------------------------------------------------------------------
    -- 🌿 PLANTUML PREVIEW
    ----------------------------------------------------------------------
    { "tyru/open-browser.vim",            cmd = "OpenBrowser" },
    { "aklt/plantuml-syntax",             ft = "plantuml" },
    { "weirongxu/plantuml-previewer.vim", ft = "plantuml" },

    ----------------------------------------------------------------------
    -- 🤖 LLM: Chat with Qwen
    ----------------------------------------------------------------------
    {
        "Kurama622/llm.nvim",
        dependencies = {
            "nvim-lua/plenary.nvim",
            "MunifTanjim/nui.nvim",
        },
        keys = {
            { "<leader>ac", "<cmd>LLMSessionToggle<cr>", mode = "n", desc = "Toggle LLM Chat" },
        },
        config = function()
            require("llm").setup({
                model = "qwen-max",
                url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation",
                api_type = "openai",
                -- api_key = os.getenv("DASHSCOPE_API_KEY"), -- Uncomment if using env var
            })
        end,
    },

    ----------------------------------------------------------------------
    -- 🐞 TAGS & INDEXING
    ----------------------------------------------------------------------
    {
        "ludovicchabant/vim-gutentags",
        ft = { "c", "cpp", "java", "go", "python" },
    },

    {
        'folke/which-key.nvim',
        event = 'VeryLazy',
        opts = {
            plugins = {
                marks = true,
                registers = true,
                spelling = true,
                presets = {
                    operators = true,
                    motions = true,
                    textobjects = true,
                    windows = true,
                    nav = true,
                    trouble = true,
                },
            },
            layout = {
                spacing = 5,
                align = "left",
            },
            -- Optional: show all keymaps in one place
            on_setup_done = function()
                require('which-key').register({
                    ['<leader>'] = { name = '+leader' },
                    ['<leader>h'] = { name = '+help' },
                    ['<leader>f'] = { name = '+file' },
                    -- Add more as needed
                })
            end,
        },
    }
}
