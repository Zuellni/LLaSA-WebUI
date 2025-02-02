const form = document.querySelector("form")
const address = form.querySelector("#address")
const connect = form.querySelector("#connect")

const input = form.querySelector("#input")
const voice = form.querySelector("#voice")

const format = form.querySelector("#format")
const maxLen = form.querySelector("#maxLen")
const repetitionPenalty = form.querySelector("#repetitionPenalty")
const sampleRate = form.querySelector("#sampleRate")
const temperature = form.querySelector("#temperature")
const topK = form.querySelector("#topK")
const topP = form.querySelector("#topP")

const generate = form.querySelector("#generate")
const audio = form.querySelector("audio")

connect.addEventListener("click", async () => {
    try {
        connect.value = "Connecting..."
        const response = await fetch(`${address.value}/settings`)
        const data = await response.json()

        maxLen.value = parseInt(data.max_len)
        repetitionPenalty.value = parseFloat(data.repetition_penalty)
        sampleRate.value = parseInt(data.sample_rate)
        temperature.value = parseFloat(data.temperature)
        topK.value = parseInt(data.top_k)
        topP.value = parseFloat(data.top_p)

        for (const entry of data.audio) {
            const option = document.createElement("option")
            option.value = entry
            option.textContent = entry
            voice.append(option)
        }

        for (const entry of data.format) {
            const option = document.createElement("option")
            option.value = entry
            option.textContent = entry
            format.append(option)
        }

        connect.value = "Connected"
    } catch (error) {
        connect.value = "Error"
        console.error(error)
    }
})

form.addEventListener("submit", async (event) => {
    event.preventDefault()
    generate.value = "Generating..."

    const data = new FormData(form)
    const obj = Object.fromEntries(data.entries())

    try {
        const response = await fetch(`${address.value}/generate`, {
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