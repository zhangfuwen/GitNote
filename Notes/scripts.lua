M={}
function list_files_in_dir(dir)
  local files = {}
  local handle = vim.uv.fs_scandir(dir)
  if not handle then return files end

  repeat
    local name, type = vim.uv.fs_scandir_next(handle)
    if not name then break end
    if type == "file" and name:match("%.md$") then
      table.insert(files, dir .. "/" .. name)
    end
  until not name

  return files
end 

function list_files_recursive(dir)
  local files = {}

  local function scan(path)
    local handle = vim.uv.fs_scandir(path)
    if not handle then return end

    repeat
      local name, type = vim.uv.fs_scandir_next(handle)
      if not name then break end

      local full_path = path .. "/" .. name

      if type == "directory" then
        scan(full_path) -- recurse
      elseif type == "file" and name:match("%.md$") then
        table.insert(files, full_path)
      end
    until not name
  end

  scan(dir)
  return files
end

function read_file_lines(filepath)
  local file = io.open(filepath, "r")
  if not file then return nil end
  local lines = {}
  for line in file:lines() do
    table.insert(lines, line)
  end
  file:close()
  return lines
end

function get_markdown_title_from_file(filepath)
  local lines = read_file_lines(filepath)
  if not lines or #lines == 0 then
    return nil
  end

  -- Check if file has YAML frontmatter: starts with ---
  if lines[1] ~= "---" then
    return nil
  end

  -- Find end of frontmatter (second ---)
  local in_frontmatter = false
  local yaml_lines = {}
  local inside = false

  for i, line in ipairs(lines) do
    if i == 1 and line == "---" then
      inside = true
      in_frontmatter = true
    elseif inside and line == "---" then
      inside = false
      break -- end of frontmatter
    elseif inside then
      table.insert(yaml_lines, line)
    end
  end

  if not in_frontmatter or #yaml_lines == 0 then
    return nil
  end

  -- Now parse YAML lines manually (simple key: value)
  for _, line in ipairs(yaml_lines) do
    local key, value = line:match("^%s*([^:%s]+)%s*:%s*(.+)")
    if key and key:lower() == "title" then
      -- Remove quotes if present
      value = value:match("^%s*[\"'](.+)[\"']%s*$") or value:match("^%s*(.+)%s*$")
      return value
    end
  end

  return nil
end

local function get_all_markdown_titles(folder, recursive)
  local files = recursive
    and list_files_recursive(folder)
    or list_files_in_dir(folder)

  local results = {}
  for _, filepath in ipairs(files) do
    local title = get_markdown_title_from_file(filepath) or "(no title)"
    table.insert(results, {
      filepath = filepath,
      title = title
    })
  end

  return results
end

M.get_all_markdown_file_and_titles = function(folder, recursive)
    local titles = get_all_markdown_titles(folder, recursive)
    return titles 
end

M.get_all_markdown_titles_text = function(folder, recursive)
    local titles = M.get_all_markdown_file_and_titles(folder, recursive)
    local results = {}
    for _, item in ipairs(titles) do
        local text = "["..item.title.."]("..item.filepath..")"
        table.insert(results, text)
    end
    return results 
end

-- open a new buffer using vim.api.nvim_buf_set_lines
function M.open_buffer(content)
  local bufnr = vim.api.nvim_create_buf(false, true)
--    local lines = type(content) == "string" and vim.split(content, '\n', true) or content
-- print(vim.inspect(content))
  vim.api.nvim_buf_set_lines(bufnr, 0, -1, false, content)
  return bufnr
end

-- open buffer in a new window
function M.open_buffer_in_new_window(bufnr)
  local winnr = vim.api.nvim_open_win(bufnr, true, {})
  return winnr
end

function M.open_buffer_in_centered_float(bufnr, width, height)
    -- ✅ 1. 获取屏幕尺寸
    local screen_width = vim.opt.columns:get()
    local screen_height = vim.opt.lines:get()

    -- ✅ 2. 计算居中位置
    local win_width = width or 60
    local win_height = height or 15

    --    local col = (screen_width - win_width) // 2
    --    local row = (screen_height - win_height) // 2
    local col = 30
    local row = 40
    -- print("screen_width ", screen_width)
    -- print("screen_height ", screen_height)
    -- print("win_width ", win_width)
    -- print("win_height ", win_height)

    -- ✅ 3. 创建浮动窗口
    local opts = {
        relative = "editor",
        width = win_width,
        height = win_height,
        row = row,
        col = col,
        style = "minimal",  -- 可选："border", "minimal"
        border = "rounded", -- 可选：none, single, double, rounded, solid, shadow
        noautocmd = true,
    }

    local winid = vim.api.nvim_open_win(bufnr, true, opts)

    -- ✅ 4. 设置窗口选项（可选）
    vim.wo[winid].wrap = false
    vim.wo[winid].cursorline = true
    vim.wo[winid].signcolumn = "no"

    -- ✅ 5. 自动关闭快捷键
    vim.keymap.set("n", "<leader>c", function()
        vim.api.nvim_win_close(winid, true)
    end, { desc = "Close centered float", buffer = bufnr })
end

bufnr = M.open_buffer(M.get_all_markdown_titles_text("/mnt/c/Users/刘涛/Documents/Code/GitNote/Notes/006-ai", true))
M.open_buffer_in_centered_float(bufnr)


