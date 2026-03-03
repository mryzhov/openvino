from pathlib import Path

def create_cpp_test(op_name: str, test_dir: Path):
    test_file = test_dir / f"test_{op_name}.cpp"

    if test_file.exists():
        return f"⚠️ {test_file.name} already exists"

    test_file.write_text(f"""
#include <gtest/gtest.h>

TEST({op_name}Translator, Basic) {{
    ASSERT_TRUE(true); // TODO: add real checks
}}
""")
    return f"✅ C++ test {test_file.name} created"
