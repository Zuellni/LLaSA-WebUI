const form = document.querySelector("form")
const server = form.querySelector("#server")
const connect = form.querySelector("#connect")

const input = form.querySelector("#input")
const voice = form.querySelector("#voice")

const maxLen = form.querySelector("#maxLen")
const format = form.querySelector("#format")
const rate = form.querySelector("#rate")
const reuse = form.querySelector("#reuse")
const seed = form.querySelector("#seed")
const penalty = form.querySelector("#penalty")
const temp = form.querySelector("#temp")
const topK = form.querySelector("#topK")
const topP = form.querySelector("#topP")

const generate = form.querySelector("#generate")
const audio = form.querySelector("audio")

const addOption = (node, text, value) => {
    const option = document.createElement("option")
    option.textContent = text
    option.value = value
    node.append(option)
}

connect.addEventListener("click", async () => {
    try {
        connect.value = "Connecting..."
        format.textContent = ""
        reuse.textContent = ""
        voice.textContent = ""

        addOption(reuse, "false", "false")
        addOption(reuse, "true", "true")
        addOption(voice, "none", "")

        const response = await fetch(`${server.value}/settings`)
        const data = await response.json()

        for (const entry of data.formats) {
            addOption(format, entry, entry)
        }

        for (const entry of data.voices) {
            addOption(voice, entry, entry)
        }

        format.value = data.format
        reuse.value = data.reuse
        voice.value = data.voice

        penalty.value = parseFloat(data.repetition_penalty)
        temp.value = parseFloat(data.temperature)
        topP.value = parseFloat(data.top_p)

        maxLen.value = parseInt(data.max_len)
        seed.value = parseInt(data.seed)
        rate.value = parseInt(data.sample_rate)
        topK.value = parseInt(data.top_k)

        connect.value = "Connected"
    } catch (error) {
        connect.value = "Error"
        console.error(error)
    }
})

form.addEventListener("submit", async (event) => {
    generate.value = "Generating..."
    event.preventDefault()

    const data = new FormData(form)
    const obj = Object.fromEntries(data.entries())

    try {
        const response = await fetch(`${server.value}/generate`, {
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