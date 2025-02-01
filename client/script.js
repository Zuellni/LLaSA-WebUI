const url = "http://127.0.0.1:8020"
const player = document.querySelector("audio")
const form = document.querySelector("form")

const text = form.querySelector("#text")
const audio = form.querySelector("#audio")
const format = form.querySelector("#format")
const maxLen = form.querySelector("#maxLen")
const repetitionPenalty = form.querySelector("#repetitionPenalty")
const sampleRate = form.querySelector("#sampleRate")
const temperature = form.querySelector("#temperature")
const topK = form.querySelector("#topK")
const topP = form.querySelector("#topP")

const generate = form.querySelector("#generate")
const stream = form.querySelector("#stream")

const getFormData = () => {
    const formData = new FormData(form)
    const formObject = Object.fromEntries(formData.entries())
    return JSON.stringify(formObject)
}

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

generate.addEventListener("click", async () => {
    if (!player.paused) {
        player.pause()
        return
    }

    generate.value = "Generating..."

    const response = await fetch(`${url}/generate`, {
        method: "post",
        headers: { "content-type": "application/json" },
        body: getFormData(),
    })

    const data = await response.blob()
    player.src = URL.createObjectURL(data)
    player.play()

    const link = document.createElement("a")
    link.download = `${audio.value}.${format.value}`
    link.href = player.src
    document.body.append(link)
    link.click()
    link.remove()

    generate.value = "Generate"
})

text.addEventListener("input", () => {
    text.style.height = ""
    text.style.height = `${text.scrollHeight}px`
})

stream.addEventListener("click", async () => {
    if (!player.paused) {
        player.pause()
        return
    }

    stream.value = "Streaming..."

    const response = await fetch(`${url}/stream`, {
        method: "post",
        headers: { "content-type": "application/json" },
        body: getFormData(),
    })

    const reader = response.body.getReader()

    while (true) {
        const { done, value } = await reader.read()

        if (value) {
            const data = new Blob([value])

            if (!player.paused && !player.ended) {
                await new Promise((resolve) => {
                    player.onended = resolve
                })
            }

            await new Promise((resolve) => {
                player.src = URL.createObjectURL(data)
                player.onended = resolve
                player.play()
            })
        }

        if (done) {
            break
        }
    }

    stream.value = "Stream"
})

getSettings()