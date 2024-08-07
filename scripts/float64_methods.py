def create_float64_methods(path: str) -> None:
    with open(path, 'rt') as f:
        content = f.read()
    float32_token = '// Float32 Methods\n'
    float64_token = '// Float64 Methods\n'
    float32_start = content.find(float32_token) + len(float32_token)
    if float32_start == len(float32_token) - 1:
        return
    float64_start = content.find(float64_token)
    header = content[:float32_start]
    float32_methods = content[float32_start:float64_start]
    float64_methods = float32_methods.replace('float', 'double')
    new_content = (
        header + float32_methods + '\n//Float64 Methods\n' + float64_methods + '}\n'
    )
    with open(path, 'wt') as f:
        f.write(new_content)


if __name__ == '__main__':
    import sys

    for file in sys.argv[1:]:
        create_float64_methods(file)
