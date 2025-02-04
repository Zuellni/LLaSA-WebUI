const form = document.querySelector("form")
const audio = form.querySelector("audio")
const format = form.querySelector("#format")
const generate = form.querySelector("#generate")
const input = form.querySelector("#input")
const voice = form.querySelector("#voice")
let running = false

form.addEventListener("submit", async (event) => {
    try {
        event.preventDefault()

        if (running) {
            generate.value = "Cancelling"
            return await fetch("cancel")
        }

        generate.value = "Generating"
        running = true

        const data = new FormData(form)
        const obj = Object.fromEntries(data.entries())

        const response = await fetch("generate", {
            method: "post",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(obj),
        })

        const blob = await response.blob()
        audio.src = URL.createObjectURL(blob)
        audio.title = `${voice.value}.${format.value}`
        audio.play()

        generate.value = "Finished"
    } catch {
        generate.value = "Error"
        console.error(error)
    } finally {
        running = false
    }
})

input.addEventListener("input", () => {
    input.style.height = ""
    input.style.height = `${input.scrollHeight}px`
})