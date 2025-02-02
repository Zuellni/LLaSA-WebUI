const url = "http://127.0.0.1:8020"
const player = document.querySelector("audio")
const form = document.querySelector("form")
const button = form.querySelector("#generate")

const text = form.querySelector("#text")
const audio = form.querySelector("#audio")
const format = form.querySelector("#format")
const maxLen = form.querySelector("#maxLen")
const repetitionPenalty = form.querySelector("#repetitionPenalty")
const sampleRate = form.querySelector("#sampleRate")
const temperature = form.querySelector("#temperature")
const topK = form.querySelector("#topK")
const topP = form.querySelector("#topP")

const getSettings = async () => {
    const response = await fetch(`${url}/settings`)
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
        audio.append(option)
    }

    for (const entry of data.format) {
        const option = document.createElement("option")
        option.value = entry
        option.textContent = entry
        format.append(option)
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault()
    button.value = "Generating..."

    const data = new FormData(form)
    const obj = Object.fromEntries(data.entries())

    try {
        const response = await fetch(`${url}/generate`, {
            method: "post",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(obj),
        })

        const blob = await response.blob()
        player.src = URL.createObjectURL(blob)
        player.title = `${audio.value}.${format.value}`
        player.play()

        button.value = "Generate"
    } catch {
        button.value = "Error"
    }
})

text.addEventListener("input", () => {
    text.style.height = ""
    text.style.height = `${text.scrollHeight}px`
    text.style.maxHeight = `${window.innerHeight - form.offsetHeight}px`
})

getSettings()