const audio = document.querySelector("audio")
const form = document.querySelector("form")
const textarea = document.querySelector("textarea")

const format = document.querySelector("#format")
const voice = document.querySelector("#voice")
const picker = document.querySelector("#picker i")
const cache = document.querySelector("#cache")

const player = document.querySelector("#player")
const submit = document.querySelector("#submit i")

const play = document.querySelector("#play i")
const timeline = document.querySelector("#timeline")
const current = document.querySelector("#current")
const duration = document.querySelector("#duration")
const volume = document.querySelector("#volume")
const download = document.querySelector("#download a")

let generating = false

const formatTime = (time) => {
    const minutes = Math.floor(time / 60).toString().padStart(2, "0")
    const seconds = Math.floor(time % 60).toString().padStart(2, "0")
    return `${minutes}:${seconds}`
}

const addOption = (node, text, value) => {
    const option = document.createElement("option")
    option.textContent = text
    option.value = value
    node.append(option)
    return option
}

form.addEventListener("submit", async (event) => {
    try {
        event.preventDefault()
        submit.textContent = "sync"

        if (generating) {
            return await fetch("abort", {
                method: "POST",
            })
        }

        submit.classList = "generating"
        generating = true

        const formData = new FormData(form)
        const obj = Object.fromEntries(formData)

        const response = await fetch("generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(obj),
        })

        const data = await response.blob()
        const url = URL.createObjectURL(data)

        audio.src = url
        audio.play()

        const name = voice.value || "random"
        download.download = `${name}.${format.value}`
        download.href = url
    } catch (error) {
        submit.textContent = "xmark-large"
        console.error(error)
    } finally {
        submit.classList = ""
        generating = false
    }
})

textarea.addEventListener("input", () => {
    textarea.style.height = ""
    textarea.style.height = `${textarea.scrollHeight}px`
})

cache.addEventListener("change", async (event) => {
    try {
        const files = event.target.files

        if (!files) {
            picker.textContent = "folder"
            return
        }

        const formData = new FormData()
        formData.append("file", files[0])

        const response = await fetch("cache", {
            method: "POST",
            body: formData,
        })

        data = await response.json()
        voice.textContent = ""
        addOption(voice, "None", "")

        for (const entry of data) {
            addOption(voice, entry, entry)
        }

        picker.textContent = "check"
    } catch (error) {
        picker.textContent = "xmark-large"
    }
})

audio.addEventListener("play", () => {
    play.textContent = "pause"
})

audio.addEventListener("pause", () => {
    play.textContent = "play"
})

audio.addEventListener("ended", () => {
    play.textContent = "play"
})

audio.addEventListener("loadedmetadata", () => {
    duration.textContent = formatTime(audio.duration)
    download.href = audio.src
})

audio.addEventListener("timeupdate", () => {
    if (!audio.duration) {
        return
    }

    const progress = audio.currentTime / audio.duration
    current.textContent = formatTime(audio.currentTime)
    timeline.setAttribute("max", "1.0")
    timeline.value = progress
})

play.addEventListener("click", () => {
    if (!audio.duration) {
        return
    }

    if (audio.paused) {
        play.textContent = "pause"
        audio.play()
    } else {
        play.textContent = "play"
        audio.pause()
    }
})

timeline.addEventListener("input", (event) => {
    if (!audio.duration) {
        return
    }

    audio.currentTime = event.target.value * audio.duration
    timeline.setAttribute("max", "1.0")
})

volume.addEventListener("input", (event) => {
    audio.volume = event.target.value
})