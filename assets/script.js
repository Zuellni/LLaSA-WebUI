const form = document.querySelector("form")
const audio = form.querySelector("audio")
const format = form.querySelector("#format")
const input = form.querySelector("#input")
const voice = form.querySelector("#voice")

form.addEventListener("submit", async (event) => {
    generate.value = "Generating..."
    event.preventDefault()

    const data = new FormData(form)
    const obj = Object.fromEntries(data.entries())

    try {
        const response = await fetch("/generate", {
            method: "post",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(obj),
        })

        const blob = await response.blob()
        audio.src = URL.createObjectURL(blob)
        audio.title = `${voice.value}.${format.value}`
        audio.play()

        generate.value = "Generate"
    } catch {
        generate.value = "Error"
        console.error(error)
    }
})

input.addEventListener("input", () => {
    input.style.height = ""
    input.style.height = `${input.scrollHeight}px`
})