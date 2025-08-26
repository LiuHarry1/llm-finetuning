def convert_to_facts(memories):
    if not memories:
        return ""
    output_lines = []
    output_lines.append("# These are the most relevant facts and their valid date ranges")
    output_lines.append("# format: FACT (Date range: from - to)")
    output_lines.append("<FACTS>")

    # 添加每个记忆项
    for memory in memories:
        content = memory.get('memory', '')
        created_at = memory.get('created_at')
        created_at = memory.get('updated_at', created_at)
        expiration_date = "present"
        if memory.get("expiration_date") and memory.get("expiration_date")!=None:
            expiration_date =memory.get("expiration_date")

        # 格式化时间范围
        if created_at:
            time_range = f"({created_at} - {expiration_date})"
        else:
            time_range = "(unknown date range)"

        output_lines.append(f"  - {content} {time_range}")

    output_lines.append("</FACTS>")

    return "\n".join(output_lines)