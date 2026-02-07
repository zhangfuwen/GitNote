-- ~/.config/nvim/lua/plugins.lua

function setup_quickui_menu()
    vim.cmd [[
call quickui#menu#reset()
call quickui#menu#install('&Find', [
\ ["Switch &Header/Source\tta", 'FSHere'],
\ ["--", '' ],
\ ["&Global search", 'FzfLua global' ],
\ ["&Search in buffer", 'FzfLua blines' ],
\ ["&Search in all buffer", 'FzfLua lines' ],
\ ["&Search in folder(live_grep)", 'FzfLua live_grep' ],
\ ["--", '' ],
\ ["Find &Files", 'FzfLua files' ],
\ ["Find &Symbols", 'FzfLua lsp_live_workspace_symbols' ],
\ ["Find &References", 'FzfLua lsp_references' ],
\ ["--", '' ],
\ ["E&xit\tAlt+x", 'echo 6' ],
\])
call quickui#menu#install('&Code', [
\ ["&Format file", ':lua vim.lsp.buf.format()'],
\ ["&Comment lines", ':lua vim.lsp.buf.execute_command({ command = "editor.action.commentLine" })'],
\ ["Cod all fold\tzR", 'normal zR'],
\ ["Generate &Doxygen comment", ':Dox'],
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
call quickui#menu#install('&Tools', [
\ ["----------\tCMake\t--------", '' ],
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
\ ["--------Git\t-----------", '' ],
\ [ "&Status\t G", 'G' ],
\ [ "&Llog\t Gllog", 'Gllog' ],
\ [ "&Clog\t Gclog", 'Gclog' ],
\ ["-------Terminal\t-------", '' ],
\ [ '&Terminal', "call quickui#terminal#open('bash', {'title':'terminal'})", 'help 1' ],
\ ["-------LLM\t-------", '' ],
\ [ "&Chat", "LLMSessionToggle"],
\ [ "&ExplainCode", "LLMSelectedTextHandler 'explain the code'"],
\ ["-------Plantuml\t-------", '' ],
\ ["&PlantumlOpen", 'PlantumlOpen' ],
\ ])

call quickui#menu#install('&Run', [
\ [ "Run this file with &python3", ":exec '!python3' shellescape(@%, 1)" ],
\ [ "Run this file with &bash", ":exec '!bash' shellescape(@%, 1)" ],
\ [ "Run codeblock(markdown)", ":call org#main#runCodeBlock()" ],
\ [ "Run language(all blocks)", ":call org#main#runLanguage()" ],
\ ])

" list
call quickui#menu#install('&Navigate', [
\ [ "&Functions", "call quickui#tools#list_function()" ],
\ [ "Buffer Tags", "FzfLua btags" ],
\ [ "&Buffers", "call quickui#tools#list_buffer('e')" ],
\ [ "&Tabs", "FzfLua tabs" ],
\ [ "&Marks", "FzfLua marks" ],
\ [ "&Jumps", "FzfLua jumps" ],
\ [ "&Changes", "FzfLua changes" ],
\ [ "Next change", "normal! g," ],
\ [ "Prev change", "normal! g;" ],
\ [ "Next jump", "normal! <c-o>" ],
\ [ "Prev jump", "normal! <c-i>" ],
\ ])

" items containing tips, tips will display in the cmdline
" script inside %{...} will be evaluated and expanded in the string
call quickui#menu#install("&Option", [
\ ["&Commands", "FzfLua commands" ],
\ ['&Colorschemes', 'FzfLua colorschemes'],
\ ['Set &Spell %{&spell? "Off":"On"}', 'set spell!'],
\ ['Set &Cursor Line %{&cursorline? "Off":"On"}', 'set cursorline!'],
\ ['Set &Paste %{&paste? "Off":"On"}', 'set paste!'],
\ ])


call quickui#menu#install('&Help', [
\ ["Edit &init.lua", 'e ~/.config/nvim/init.lua' ],
\ ["Edit &plugins.lua", 'e ~/.config/nvim/lua/plugins.lua' ],
\ ["&Lazy(plugin store)", 'Lazy', ''],
\ ["&Mason(lsp store)", 'Mason', ''],
\ ["&Keymaps", 'FzfLua keymaps', ''],
\ ["&Cheatsheet", 'help index', ''],
\ ['T&ips', 'help tips', ''],
\ ['--',''],
\ ["&Tutorial", 'help tutor', ''],
\ ['&Quick Reference', 'help quickref', ''],
\ ['&Summary', 'help summary', ''],
\ ], 10000)

]]
end

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
    { -- it is a colorscheme, with better cousor line color
        'folke/tokyonight.nvim',
        lazy = false,
        priority = 1000,
        opts = {
            -- your options
        },
        config = function(_, opts)
            require('tokyonight').setup(opts)
            vim.cmd("highlight CursorLine guibg=#1e2129 ctermbg=238")
        end,
    },

    ----------------------------------------------------------------------
    -- 📁 FILE & PROJECT MANAGEMENT
    ----------------------------------------------------------------------
    {
        'preservim/nerdtree',
        dependencies = {
            'tiagofumo/vim-nerdtree-syntax-highlight',
        },
        config = function()
            -- Load NerdTree
            vim.g.NERDTreeShowHidden = 1
            vim.g.NERDTreeAutoCenter = 1
            vim.g.NERDTreeQuitOnOpen = 1
        end,
        keys = { { "tf", "<cmd>NERDTreeToggle<cr>", desc = "Toggle NERDTree" } },
    },
    {
        'tiagofumo/vim-nerdtree-syntax-highlight',
    },
    {
        "majutsushi/tagbar",
        cmd = "TagbarToggle",
        keys = { { "tb", ":TagbarToggle<CR>", desc = "Toggle Tagbar" } },
        keys = { { "tl", ":TagbarToggle<CR>", desc = "Toggle Tagbar" } },
    },

    ----------------------------------------------------------------------
    -- 🔍 SEARCH & FIND
    ----------------------------------------------------------------------
    {
        "skywind3000/vim-quickui",
        event = "VeryLazy",
        config = function()
            vim.g.quickui_border_style = 2
            vim.keymap.set('n', 'tt', ':call quickui#menu#open()<CR>')
            vim.g.quickui_show_tip = 1
            setup_quickui_menu()
            --            vim.cmd(quickui_menu_cmds)
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
    --    { "kana/vim-textobj-user",      event = "VeryLazy" },
    --  { "kana/vim-textobj-indent",        event = "VeryLazy" },
    --  { "kana/vim-textobj-syntax",        event = "VeryLazy" },
    --    {
    --        "kana/vim-textobj-function",
    --        ft = { "c", "cpp", "vim", "java" },
    --    },
    --   { "sgur/vim-textobj-parameter", ft = { "c", "cpp", "go", "rust" } },

    ----------------------------------------------------------------------
    -- 💬 COMMENTS & AUTO-CLOSING
    ----------------------------------------------------------------------
    -- { "scrooloose/nerdcommenter",   event = "BufReadPre" },
    {
        'numToStr/comment.nvim',
        event = 'VeryLazy',
        opts = {
            -- Optional: customize behavior
            marker = {
                inline = '',
                block = '█',
            },
            toggler = {
                line = 'gc',
                block = 'gb',
            },
            mappings = {
                basic = true,
                extra = true,
            },
        },
    },
    {
        'windwp/nvim-autopairs',
        event = 'InsertEnter',
        opts = {
            check_ts = true,                    -- Use treesitter to avoid closing in comments/strings
            disable_filetype = { "TelescopePrompt", "spectre_panel" },
            ignored_next_char = [=[[%w%.%s]]=], -- Don't auto-close after these chars
            fast_wrap = {
                map = "<M-e>",
                chars = { "{", "[", "(", '"', "'" },
                pattern = [=[[%(%[%{%"%']]=],
                end_key = "e",
                keys = "qwertyuiopzxcvbnm",
                check_comma = true,
                search_namespace = "nvim_autopairs_search",
            },
        },
        config = function(_, opts)
            local autopairs = require("nvim-autopairs")
            autopairs.setup(opts)

            -- Auto setup for LSP
            local cmp_autopairs = require("nvim-autopairs.completion.cmp")
            local cmp = require("cmp")
            cmp.event:on("confirm_done", cmp_autopairs.on_confirm_done({ map_char = { tex = "" } }))
        end,
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

            --vim.keymap.set({ 'n', 'x' }, '<Leader>f', '<Plug>(easymotion-bd-f)')
            --vim.keymap.set('n', '<Leader>f', '<Plug>(easymotion-overwin-f)')
            --vim.keymap.set('n', 's', '<Plug>(easymotion-overwin-f2)')
            --vim.keymap.set({ 'n', 'x' }, '<C-L>', '<Plug>(easymotion-bd-jk)')
            --vim.keymap.set('n', '<C-L>', '<Plug>(easymotion-overwin-line)')
            --vim.keymap.set({'n', 'x'}, '<C-L>u', '<Plug>(easymotion-bd-w)')
            --vim.keymap.set('n', '<C-L>u', '<Plug>(easymotion-overwin-w)')
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
        version = "v2.*",
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
            require("luasnip.loaders.from_snipmate").lazy_load()
        end,
    },

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
        opts = function()
            local cmp = require("cmp")
            return {
                -- Enable auto-selection of first item
                completion = {
                    completeopt = "menuone,noinsert",
                },
            }
        end,
        config = function()
            local cmp = require("cmp")
            local luasnip = require("luasnip")
            cmp.setup({
                snippet = {
                    expand = function(args)
                        require("luasnip").lsp_expand(args.body)
                    end,
                },
                mapping = cmp.mapping.preset.insert({

                    ["<C-Space>"] = cmp.mapping.complete(),
                    ["<C-b>"] = cmp.mapping.scroll_docs(-4),
                    ["<C-f>"] = cmp.mapping.scroll_docs(4),
                    ["<CR>"] = cmp.mapping.confirm({ select = false }),
                    ["<Tab>"] = cmp.mapping(function(fallback)
                        if cmp.visible() then
                            cmp.select_next_item()
                        elseif luasnip.expand_or_jumpable() then
                            luasnip.expand_or_jump()
                        -- elseif has_words_before() then
                        --     cmp.complete()
                        else
                            fallback()
                        end
                    end, { "i", "s" }),
                    ["<S-Tab>"] = cmp.mapping(function(fallback)
                        if cmp.visible() then
                            cmp.select_prev_item()
                        elseif luasnip.jumpable(-1) then
                            luasnip.jump(-1)
                        else
                            fallback()
                        end
                    end, { "i", "s" }),
                    -- ["<Tab>"] = cmp.mapping(function(fallback)
                    --     if cmp.visible() then
                    --         cmp.select_next_item()
                    --     else
                    --         fallback()
                    --     end
                    -- end, { "i", "s" }),
                    -- ["<S-Tab>"] = cmp.mapping(function(fallback)
                    --     if cmp.visible() then
                    --         cmp.select_prev_item()
                    --     else
                    --         fallback()
                    --     end
                    -- end, { "i", "s" }),
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
    { "airblade/vim-gitgutter", event = "BufReadPre" },
    { "tpope/vim-fugitive",     cmd = { "Git", "G" } },
    { "will133/vim-dirdiff",    cmd = "DirDiff" },
    { "gregsexton/gitv",        cmd = "Gitv" },

    ----------------------------------------------------------------------
    -- 🧱 BUILD / PROJECT TOOLS
    ----------------------------------------------------------------------
    { "ilyachur/cmake4vim",     ft = "cmake" },

    ----------------------------------------------------------------------
    -- 🌿 PLANTUML PREVIEW
    ----------------------------------------------------------------------
    { "tyru/open-browser.vim",  cmd = "OpenBrowser" },
    { "aklt/plantuml-syntax",   ft = "plantuml" },
    {
        "weirongxu/plantuml-previewer.vim",
        ft = "plantuml",
        dependencies = {
            "tyru/open-browser.vim"
        }
    },

    ----------------------------------------------------------------------
    -- 🤖 LLM: Chat with Qwen
    ----------------------------------------------------------------------
    {
        "Kurama622/llm.nvim",
        dependencies = {
            "nvim-lua/plenary.nvim",
            "MunifTanjim/nui.nvim",
        },
        cmd = { "LLMSessionToggle", "LLMSelectedTextHandler", "LLMAppHandler" },
        keys = {
            { "<leader>ac", "<cmd>LLMSessionToggle<cr>", mode = "n", desc = "Toggle LLM Chat" },
            { "<leader>ae", mode = "v", "<cmd>LLMSelectedTextHandler 请解释下面这段代码<cr>", desc = "explain code(LLM)" },
            { "<leader>ts", mode = "x", "<cmd>LLMSelectedTextHandler 英译汉<cr>", desc = "translate(LLM)" },
            { "<leader>ar", mode = "n", "<cmd>LLMAppHandler BashRunner<cr>", desc = "bash runner" },
        },
        config = function()
            local llm = require("llm")
            local tools = require("llm.tools")
            print(vim.inspect(llm))
            llm.setup({
                model = "qwen3-30b-a3b-instruct-2507",
                url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                api_type = "openai",
                -- api_key = os.getenv("DASHSCOPE_API_KEY"), -- Uncomment if using env var
                app_handler = {
                    BashRunner = {
                        handler = tools.qa_handler,
                        prompt = [[Write a suitable bash script and run it through CodeRunner]],
                        opts = {
                            model = "qwen3-30b-a3b-instruct-2507",
                            url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                            api_type = "openai",
                            max_tokens = 4096,
                            enable_thinking = false,

                            component_width = "60%",
                            component_height = "50%",
                            query = {
                                title = "  CodeRunner ",
                                hl = { link = "Define" },
                            },
                            input_box_opts = {
                                size = "15%",
                                win_options = {
                                    winhighlight = "Normal:Normal,FloatBorder:FloatBorder",
                                },
                            },
                            preview_box_opts = {
                                size = "85%",
                                win_options = {
                                    winhighlight = "Normal:Normal,FloatBorder:FloatBorder",
                                },
                            },
                            functions_tbl = {
                                CodeRunner = function(code)
                                    print("try running code")
                                    local filepath = "/tmp/script.sh"

                                    -- Print the code suggested by llm
                                    vim.notify(
                                        string.format("CodeRunner running...\n```bash\n%s\n```", code),
                                        vim.log.levels.INFO,
                                        { title = "llm: CodeRunner" }
                                    )

                                    local file = io.open(filepath, "w")
                                    if file then
                                        file:write(code)
                                        file:close()
                                        local script_result = vim.system({ "bash", filepath }, { text = true }):wait()
                                        os.remove(filepath)
                                        return script_result.stdout
                                    else
                                        return ""
                                    end
                                end,
                            },
                            schema = {
                                {
                                    type = "function",
                                    ["function"] = {
                                        name = "CodeRunner",
                                        description = "Bash code interpreter",
                                        parameters = {
                                            properties = {
                                                code = {
                                                    type = "string",
                                                    description = "bash code",
                                                },
                                            },
                                            required = { "code" },
                                            type = "object",
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
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
        "Kurama622/markdown-org",
        ft = "markdown",
        config = function()
            vim.g.language_path = {
                python = "python",
                python3 = "python3",
                go = "go",
                c = "gcc -Wall",
                cpp = "g++ -std=c++17 -Wall",
                bash = "bash",
                ["c++"] = "g++ -std=c++17 -Wall",
            }
            return {
                default_quick_keys = 0,
                vim.api.nvim_set_var("org#style#border", 2),
                vim.api.nvim_set_var("org#style#bordercolor", "FloatBorder"),
                vim.api.nvim_set_var("org#style#color", "String"),
            }
        end,
        keys = {
            { "<leader>mr", "<cmd>call org#main#runCodeBlock()<cr>" },
            { "<leader>ml", "<cmd>call org#main#runLanguage()<cr>" },
        },
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
    },

}
