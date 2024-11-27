def filter_bad_words(text):
    # Lista de palabras malas
    bad_words = ['mala palabra 1', 'mala palabra 2', 'mala palabra 3']  # Añadir más palabras según sea necesario
    for word in bad_words:
        text = text.replace(word, '*' * len(word))  # Reemplaza las palabras malas por asteriscos
    return text

# Prueba de ejemplo
input_text = "Este es un ejemplo con mala palabra 1"
filtered_text = filter_bad_words(input_text)
print(filtered_text)  # Salida: "Este es un ejemplo con ***** ********"
