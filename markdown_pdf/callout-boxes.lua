local environments = {
  questionbox = "questionbox",
  infobox = "infobox",
  bluebox = "infobox",
  warningbox = "warningbox",
  redbox = "warningbox",
  successbox = "successbox",
  greenbox = "successbox",
}

local function escape_latex(text)
  local replacements = {
    ["\\"] = "\\textbackslash{}",
    ["{"] = "\\{",
    ["}"] = "\\}",
    ["#"] = "\\#",
    ["$"] = "\\$",
    ["%"] = "\\%",
    ["&"] = "\\&",
    ["_"] = "\\_",
    ["^"] = "\\textasciicircum{}",
    ["~"] = "\\textasciitilde{}",
  }
  return (text:gsub("[\\{}#$%%&_^~]", replacements))
end

function Div(element)
  local environment = nil
  for _, class in ipairs(element.classes) do
    if environments[class] then
      environment = environments[class]
      break
    end
  end
  if not environment then
    return nil
  end

  local title = element.attributes.title or ""
  local opening = "\\begin{" .. environment .. "}"
  if title ~= "" and environment ~= "questionbox" then
    opening = opening .. "[" .. escape_latex(title) .. "]"
  end

  local blocks = {pandoc.RawBlock("latex", opening)}
  for _, block in ipairs(element.content) do
    table.insert(blocks, block)
  end
  table.insert(blocks, pandoc.RawBlock("latex", "\\end{" .. environment .. "}"))
  return blocks
end
